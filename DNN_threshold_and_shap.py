# -*- coding: utf-8 -*-
"""
DNN_threshold_and_shap_fast.py
--------------------------------------
Speed-optimized post-hoc analyzer.

- Threshold sweep on ALL preprocessed splits (train+val+test) -> GPU if available.
- Fast SHAP summary:
  * CPU KernelExplainer
  * background summarized via shap.kmeans(..., K=30)
  * explain only up to 2000 samples
- Stable sigmoid to prevent overflow.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

IN_CSV     = "dnn_ready_dataset.csv"
MODEL_PTH  = "DNN_best_model.pth"
THRESH_PNG = "DNN_threshold_optimization.png"
SHAP_PNG   = "DNN_shap_summary.png"


# ---------- Utils ----------
def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid using clipping to avoid overflow."""
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def _map_feature_labels(feat_names):
    """
    Replace categorical codes with human-friendly labels for SHAP.
    - Aspect codes (1-9) -> compass directions (EN)
    - Land use codes -> English land cover types
    """
    aspect_map = {
        "1": "North",
        "2": "North-East",
        "3": "East",
        "4": "South-East",
        "5": "South",
        "6": "South-West",
        "7": "West",
        "8": "North-West",
        "9": "Unexposed",
    }
    landuse_map = {
        "1": "Water bodies",
        "2": "Forests",
        "4": "Seasonally flooded areas",
        "5": "Agricultural land",
        "7": "Settlements",
        "8": "Barren surfaces",
        "9": "Snow / Ice",
        "10": "Clouds (no data)",
        "11": "Pastures / Grasslands",
    }

    def _rename(name):
        n = name.replace(" ", "")
        m = re.match(r"(?i)Aspect(?:==|=)(\d+)", n)
        if m:
            return f"Aspect: {aspect_map.get(m.group(1), m.group(1))}"
        m = re.match(r"(?i)Land[_\s]*use(?:==|=)(\d+)", n)
        if m:
            return f"Land use: {landuse_map.get(m.group(1), m.group(1))}"
        return name

    return [_rename(n) for n in feat_names]


# ---------- Model (matches residual checkpoint keys) ----------
class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.15):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.bn1  = nn.BatchNorm1d(d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_out)
        self.bn2  = nn.BatchNorm1d(d_out)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.drop = nn.Dropout(p=p_drop)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.drop(self.act(self.bn1(self.lin1(x))))
        out = self.drop(self.act(self.bn2(self.lin2(out))))
        if not isinstance(self.proj, nn.Identity):
            identity = self.proj(identity)
        return out + identity


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, widths=None):
        super().__init__()
        if widths is None:
            widths = [512, 256, 128, 128]
        d = in_dim
        blocks = []
        for w in widths:
            blocks.append(ResidualBlock(d, w, w, p_drop=0.15))
            d = w
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(d, 1)

    def rebuild_from_state_dict(self, sd):
        widths = []
        i = 0
        while f"backbone.{i}.lin2.weight" in sd:
            widths.append(sd[f"backbone.{i}.lin2.weight"].shape[0])
            i += 1
        if i > 0:
            in_dim = sd[f"backbone.0.lin1.weight"].shape[1]
            self.__init__(in_dim, widths=widths)

    def forward(self, x):
        return self.head(self.backbone(x))


@torch.no_grad()
def logits_on_array(model, X_np, device, batch_size=8192):
    """Batch inference over a large numpy array."""
    ds = TensorDataset(torch.from_numpy(X_np.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    outs = []
    for (xb,) in loader:
        xb = xb.to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0).ravel()


def _load_checkpoint(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _load_all_preprocessed():
    """Concatenate preprocessed train/val/test splits (npy)."""
    partsX, partsY = [], []
    for sp in ("train", "val", "test"):
        x_path = f"dnn_X_{sp}.npy"
        y_path = f"dnn_y_{sp}.npy"
        if os.path.exists(x_path) and os.path.exists(y_path):
            partsX.append(np.load(x_path))
            partsY.append(np.load(y_path).astype(int).ravel())
    if not partsX:
        raise FileNotFoundError("No preprocessed splits found (dnn_X_* / dnn_y_*).")
    return np.concatenate(partsX, axis=0), np.concatenate(partsY, axis=0)


# ---------- Threshold sweep (GPU) ----------
def run_threshold_all():
    X, y = _load_all_preprocessed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = _load_checkpoint(MODEL_PTH)

    # IMPORTANT: rebuild on CPU, then move to device AFTER rebuild, BEFORE inference
    model = NeuralNetwork(in_dim=X.shape[1])   # create on CPU
    model.rebuild_from_state_dict(sd)          # rebuild layers (still CPU)
    model.load_state_dict(sd, strict=False)    # load weights on CPU
    model.to(device)                           # move full model to GPU/CPU (fixes device mismatch)

    logits = logits_on_array(model, X, device=device, batch_size=8192)
    probs = _stable_sigmoid(logits)

    ts = np.linspace(0.0, 1.0, 101)
    accs, f1s = [], []
    from sklearn.metrics import f1_score
    for t in ts:
        pred = (probs > t).astype(int)
        accs.append((pred == y).mean() * 100.0)
        f1s.append(f1_score(y, pred) * 100.0)

    accs = np.asarray(accs); f1s = np.asarray(f1s)
    best_idx = int(np.argmax(f1s))
    best_t, best_f1 = ts[best_idx], f1s[best_idx]

    plt.figure(figsize=(12, 7))
    plt.plot(ts, accs, label="Accuracy (%)", linewidth=2.2)
    plt.plot(ts, f1s, label="F1 Score (%)", linewidth=2.2)
    plt.axvline(best_t, color="gray", linestyle="--", linewidth=1.2)
    plt.scatter([best_t], [best_f1], s=70)

    # More aggressive left shift when near the right edge
    x_off = -0.25 if best_t > 0.90 else 0.03
    y_off = -5 if best_t > 0.90 else 3
    plt.annotate(
        f"Best F1: {best_f1:.1f}% @ {best_t:.2f}",
        xy=(best_t, best_f1),
        xytext=(best_t + x_off, best_f1 + y_off),
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )

    plt.title("DNN - Threshold Optimization")
    plt.xlabel("Threshold"); plt.ylabel("Metric (%)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(THRESH_PNG, dpi=300); plt.close()
    print(f"[OK] Saved threshold sweep figure -> {THRESH_PNG} | Best F1 = {best_f1:.2f}% @ t={best_t:.3f}")


# ---------- Fast SHAP (CPU, k-means background) ----------
def run_shap_fast(K=30, explain_n=2000):
    if not _HAS_SHAP:
        print("[WARN] shap not installed; skipping SHAP.")
        return
    try:
        X, _ = _load_all_preprocessed()
    except Exception as e:
        print(f"[WARN] {e}; skipping SHAP.")
        return

    feat_dim = X.shape[1]

    # Summarize background with k-means (fast & stable)
    try:
        background = shap.kmeans(X, K)
    except Exception:
        np.random.seed(42)
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=min(K, X.shape[0]), replace=False)
        background = X[idx]

    # Sample a subset to explain
    rng = np.random.default_rng(42)
    ex_idx = rng.choice(X.shape[0], size=min(explain_n, X.shape[0]), replace=False)
    X_explain = X[ex_idx].astype(np.float32)

    # CPU model wrapper for KernelExplainer
    device_cpu = torch.device("cpu")
    sd = _load_checkpoint(MODEL_PTH)
    model = NeuralNetwork(in_dim=feat_dim)     # CPU
    model.rebuild_from_state_dict(sd)
    model.load_state_dict(sd, strict=False)
    model.to(device_cpu)
    model.eval()

    def f_cpu(x_np):
        with torch.no_grad():
            xb = torch.from_numpy(x_np.astype(np.float32))
            logits = model(xb)
            return torch.sigmoid(logits).numpy().ravel()

    try:
        expl = shap.KernelExplainer(f_cpu, background)
        shap_vals = expl.shap_values(X_explain, l1_reg="aic")
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
    except Exception as e:
        print(f"[ERROR] KernelExplainer failed: {e}. Skipping SHAP.")
        return

    # --- Prepare feature names (raw + pretty) ---
    # 1) Read raw column names to keep a stable key for joins across models (CSV).
    if os.path.exists(IN_CSV):
        try:
            cols = pd.read_csv(IN_CSV, nrows=5)
            feat_names = [c for c in cols.columns if c != "label"] if "label" in cols.columns else list(cols.columns)
            if len(feat_names) != feat_dim:
                feat_names = [f"feat_{i}" for i in range(feat_dim)]
        except Exception:
            feat_names = [f"feat_{i}" for i in range(feat_dim)]
    else:
        feat_names = [f"feat_{i}" for i in range(feat_dim)]

    # 2) Keep two versions:
    feat_names_raw = feat_names[:]                    # exact names for CSV + cross-model merge
    feat_names_pretty = _map_feature_labels(feat_names)  # human-friendly labels for plots

    # === Compute mean absolute SHAP values per feature ===
    # This represents the overall importance of each feature across all explained samples.
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)  # shape: (n_features,)

    # === Build DataFrame and assign feature ranks ===
    df_shap = pd.DataFrame({
        "feature": feat_names_raw,           # raw name, used for merging across models
        "label_pretty": feat_names_pretty,   # display name for readers
        "mean_abs_shap": mean_abs_shap
    })

    # Rank features by importance (1 = most important)
    df_shap["rank"] = df_shap["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    df_shap = df_shap.sort_values("rank", kind="stable")

    # === Save SHAP results to CSV ===
    # This CSV will later be used for cross-model Spearman correlation analysis.
    df_shap.to_csv("DNN_shap_values.csv", index=False)
    print("[OK] Saved CSV with per-feature SHAP importances -> DNN_shap_values.csv")

    # === Plot SHAP summary ===
    plt.figure()
    shap.summary_plot(shap_vals, X_explain, feature_names=feat_names_pretty, show=False)
    plt.title("DNN - SHAP summary")
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved SHAP summary figure -> {SHAP_PNG} (fast: K={K}, N={X_explain.shape[0]})")


# ---------- Main ----------
if __name__ == "__main__":
    try:
        run_threshold_all()
    except Exception as e:
        print(f"[WARN] Threshold sweep failed: {e}")
    run_shap_fast(K=30, explain_n=2000)
