# -*- coding: utf-8 -*-
# KAN_Train.py — improved training with residual MLP, OneCycleLR, EMA, early-stop logs, total runtime
import os, re, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, average_precision_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

IN_CSV     = "dnn_ready_dataset.csv"
MODEL_PATH = "KAN_best_model.pth"
REPORT_TXT = "KAN_evaluation_report.txt"
CM_PNG     = "KAN_confusion_matrix.png"
META_JSON  = "KAN_train_meta.json"

# ====== Hyperparams ======
BATCH_SIZE   = 4096          # try 8192 on RTX 5090 if VRAM allows (AMP enabled)
NUM_EPOCHS   = 300           # OneCycleLR works best with 60–120 epochs
INIT_LR      = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP   = 10
AMP_ENABLED  = True
NUM_WORKERS  = 0             # keep 0 on Windows unless you're comfortable with multiprocessing

# ====== Helpers ======
def _fmt_seconds(s: float) -> str:
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# ====== Model (Residual MLP with BN + SiLU) ======
class _Block(nn.Module):
    def __init__(self, in_dim, out_dim, p_drop=0.15):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.bn1  = nn.BatchNorm1d(out_dim)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(p_drop)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.bn2  = nn.BatchNorm1d(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        y = self.lin1(x); y = self.bn1(y); y = self.act(y); y = self.drop(y)
        y = self.lin2(y); y = self.bn2(y)
        return self.act(y + self.proj(x))

class KANModel(nn.Module):
    """Residual MLP backbone (no final sigmoid; we use BCEWithLogitsLoss)."""
    def __init__(self, input_size):
        super().__init__()
        widths = [256, 128, 64]
        d = input_size
        layers = []
        for w in widths:
            layers.append(_Block(d, w, p_drop=0.15))
            d = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)  # logits
    def forward(self, x): 
        return self.head(self.backbone(x))

@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    logits_all, y_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item()
        logits_all.append(logits.cpu())
        y_all.append(yb.cpu())
    logits = torch.cat(logits_all).numpy().ravel()
    y_true = torch.cat(y_all).numpy().ravel().astype(int)
    probs = 1/(1+np.exp(-logits))
    preds = (probs > threshold).astype(int)
    acc = (preds == y_true).mean()*100
    f1  = f1_score(y_true, preds)
    try:
        pr_auc = average_precision_score(y_true, probs)
        roc    = roc_auc_score(y_true, probs)
    except Exception:
        pr_auc = float("nan"); roc = float("nan")
    return total_loss/len(loader), acc, f1, pr_auc, roc, probs, y_true

def run():
    script_start = time.time()

    # ====== Load data (already scaled + one-hot) ======
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(IN_CSV)
    df = pd.read_csv(IN_CSV)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    df["label"] = (df["label"].astype(float) > 0).astype(int)
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS>0)
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val.reshape(-1,1), dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
        num_workers=NUM_WORKERS, persistent_workers=(NUM_WORKERS>0)
    )

    # ====== Model / Loss / Optim / Scheduler ======
    model = KANModel(input_size=X_train.shape[1]).to(device)

    cw = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    pos_weight = torch.tensor(cw[1] / cw[0], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=INIT_LR,
        steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS,
        pct_start=0.15, div_factor=10.0, final_div_factor=100.0
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(AMP_ENABLED and device.type=='cuda'))

    # ====== EMA (Exponential Moving Average) of weights ======
    ema_decay = 0.999
    ema = {}
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k] = v.detach().clone()
        else:
            # keep a direct copy/reference for non-float buffers (e.g., num_batches_tracked)
            ema[k] = v

    # ====== Train loop with early-stop logs & ETA ======
    best_val_loss = float("inf")
    patience = 0
    threshold = 0.5

    print(f"Training for {NUM_EPOCHS} epochs | init LR = {INIT_LR}")
    epoch_times = []

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(AMP_ENABLED and device.type=='cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # EMA update
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)
                    else:
                        # keep buffers in sync without EMA math
                        ema[k] = v

            run_loss += loss.item()
            preds = (torch.sigmoid(logits) > threshold).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / max(1,total) * 100
        val_loss, val_acc, val_f1, val_pr, val_roc, _, _ = validate(
            model, val_loader, criterion, device, threshold
        )

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (NUM_EPOCHS - (epoch + 1))
        cur_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss {run_loss/len(train_loader):.4f} Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% F1 {val_f1:.4f} "
            f"PR-AUC {val_pr:.4f} ROC-AUC {val_roc:.4f} | "
            f"Time {elapsed:.1f}s, Rem: {_fmt_seconds(eta)}"
        )

        improved = val_loss < best_val_loss
        if improved:
            print(f"Val loss improved {best_val_loss:.4f} ? {val_loss:.4f}")
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            print(f"No improvement. Best: {best_val_loss:.4f} | LR={cur_lr:.6f} | Patience {patience}/{EARLY_STOP}")
            if patience >= EARLY_STOP:
                print("? Early stopping.")
                break

    # ====== Load EMA weights for final eval ======
    model.load_state_dict(ema, strict=False)

    # ====== Threshold tuning on validation (maximize F1) ======
    _, _, _, _, _, val_probs, val_true = validate(model, val_loader, criterion, device, threshold=0.5)
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in ts:
        f1 = f1_score(val_true, (val_probs>t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    threshold = float(best_t)
    print(f"?? Best validation threshold = {threshold:.2f} (F1={best_f1:.3f})")

    # ====== Final test ======
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_test_t).squeeze(1).cpu().numpy()
        probs  = 1/(1+np.exp(-logits))
        preds  = (probs > threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    pr_auc = average_precision_score(y_test, probs)
    roc    = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    print("\n=== Test metrics ===")
    print(f"Acc {acc:.4f} | F1 {f1:.4f} | PR-AUC {pr_auc:.4f} | ROC-AUC {roc:.4f}")
    print(classification_report(y_test, preds))

    # Confusion matrix plot
    labels = np.array([
        [f"{tn}\n({tn/total:.1%})", f"{fp}\n({fp/total:.1%})"],
        [f"{fn}\n({fn/total:.1%})", f"{tp}\n({tp/total:.1%})"]
    ])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
                xticklabels=["No Fire","Fire"], yticklabels=["No Fire","Fire"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("KAN - Confusion Matrix")
    plt.tight_layout(); plt.savefig(CM_PNG, dpi=300); plt.show()

    # === Feature "importance" (first-layer weights) ===
    try:
        with torch.no_grad():
            # get first Linear layer encountered
            first_lin = None
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    first_lin = m
                    break
            if first_lin is not None:
                importance = first_lin.weight.abs().mean(dim=0).cpu().numpy()
                feat_names = df.drop(columns=["label"]).columns
                top = int(min(20, importance.size))
                idx = np.argsort(importance)[::-1][:top]

                plt.figure(figsize=(8,6))
                plt.barh(range(top), importance[idx][::-1])
                plt.yticks(
                    range(top),
                    [re.sub(r"_resampled(_\d+)?","", feat_names[i]) for i in idx[::-1]]
                )
                plt.xlabel("Mean |Weight| (First Layer)")
                plt.title("KAN - Top Features (proxy importance)")
                plt.tight_layout()
                plt.savefig("KAN_feature_importance_gain.png", dpi=300)
                plt.close()
                print("?? Saved feature importance ? KAN_feature_importance_gain.png")
    except Exception as e:
        print("?? Feature importance plot failed:", e)

    # ====== Save report (with runtime) ======
    total_runtime = time.time() - script_start
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"Best val threshold: {threshold:.2f}\n")
        f.write(f"Test Acc {acc:.4f} | F1 {f1:.4f} | PR-AUC {pr_auc:.4f} | ROC-AUC {roc:.4f}\n\n")
        f.write("Test Classification Report:\n")
        f.write(classification_report(y_test, preds) + "\n")
        f.write("Confusion Matrix Values:\n")
        f.write(f"  TN: {tn} ({tn/total:.1%})  FP: {fp} ({fp/total:.1%})\n")
        f.write(f"  FN: {fn} ({fn/total:.1%})  TP: {tp} ({tp/total:.1%})\n\n")
        f.write(f"Total runtime: {_fmt_seconds(total_runtime)}\n")
    print(f"?? Saved: {REPORT_TXT}")
    print(f"?  Total runtime: {_fmt_seconds(total_runtime)}")

    # ====== Meta ======
    meta = {
        "input_csv": IN_CSV,
        "n_features": int(X.shape[1]),
        "batch_size": BATCH_SIZE,
        "epochs_planned": NUM_EPOCHS,
        "best_val_loss": float(best_val_loss),
        "threshold": threshold,
        "pos_weight": float(pos_weight.item()),
        "optimizer": "AdamW",
        "init_lr": INIT_LR,
        "weight_decay": WEIGHT_DECAY,
        "amp": bool(AMP_ENABLED),
        "ema_decay": ema_decay
    }
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("?? Saved meta ?", META_JSON)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()   # Windows-safe
    run()
