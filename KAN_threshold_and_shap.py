# -*- coding: utf-8 -*-
"""
KAN_threshold_and_shap.py
---------------------------------
Post-hoc analysis for the trained KAN (Residual MLP) wildfire model.

- Threshold sweep on ALL preprocessed splits (train+val+test) -> GPU if available.
- Fast SHAP summary (CPU KernelExplainer) with k-means background (K=30) and N=2000 samples.
- Feature label mapping for Aspect/Land use (incl. class 11 -> Pastures / Grasslands).
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

# ------------ Paths ------------
IN_CSV     = "dnn_ready_dataset.csv"
MODEL_PTH  = "KAN_best_model.pth"      # from KAN_Train.py
THRESH_PNG = "KAN_threshold_optimization.png"
SHAP_PNG   = "KAN_shap_summary.png"

# ------------ Utils ------------
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

# ------------ Model (matches KAN_Train.py) ------------
class _Block(nn.Module):
    """Residual block: Linear -> BN -> SiLU -> Dropout -> Linear -> BN + skip."""
    def __init__(self, in_dim, out_dim, p_drop=0.15):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.bn1  = nn.BatchNorm1d(out_dim)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(p=p_drop)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.bn2  = nn.BatchNorm1d(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        y = self.lin1(x); y = self.bn1(y); y = self.act(y); y = self.drop(y)
        y = self.lin2(y); y = self.bn2(y)
        return self.act(y + self.proj(x))

class KANModel(nn.Module):
    """Residual MLP backbone; head outputs logits (no final sigmoid)."""
    def __init__(self, input_size, widths=(256, 128, 64)):
        super().__init__()
        d = input_size
        layers = []
        for w in widths:
            layers.append(_Block(d, w, p_drop=0.15))
            d = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
    def forward(self, x):
        return self.head(self.backbone(x))

@torch.no_grad()
def _logits_on_array(model, X_np, device, batch_size=8192):
    """Batch inference for large numpy arrays."""
    ds = TensorDataset(torch.from_numpy(X_np.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    outs = []
    for (xb,) in loader:
        xb = xb.to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0).ravel()

def _load_all_preprocessed():
    """
    Concatenate preprocessed train/val/test splits if available.
    Looks for dnn_X_* (primary) then kan_X_* (secondary).
    Falls back to CSV if none are found.
    """
    def _try_stack(prefix):
        partsX, partsY = [], []
        for sp in ("train", "val", "test"):
            x_path = f"{prefix}_X_{sp}.npy"
            y_path = f"{prefix}_y_{sp}.npy"
            if os.path.exists(x_path) and os.path.exists(y_path):
                partsX.append(np.load(x_path))
                partsY.append(np.load(y_path).astype(int).ravel())
        if partsX:
            return np.concatenate(partsX, 0), np.concatenate(partsY, 0)
        return None, None

    X, y = _try_stack("dnn")
    if X is None:
        X, y = _try_stack("kan")
    if X is not None:
        return X, y

    # Fallback: load entire CSV
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError("No preprocessed arrays found and CSV is missing.")
    df = pd.read_csv(IN_CSV)
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column.")
    y = df["label"].to_numpy().astype(int).ravel()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    return X, y

def _load_feature_names(n_feats: int):
    """Pull feature names from CSV; fallback to generic if length mismatch."""
    if os.path.exists(IN_CSV):
        try:
            head = pd.read_csv(IN_CSV, nrows=5)
            names = [c for c in head.columns if c != "label"] if "label" in head.columns else list(head.columns)
            if len(names) == n_feats:
                return names
        except Exception:
            pass
    return [f"feat_{i}" for i in range(n_feats)]

# ------------ Threshold sweep (GPU) ------------
def run_threshold_all():
    X, y = _load_all_preprocessed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model with correct input size; load weights
    model = KANModel(input_size=X.shape[1])   # create on CPU
    state = torch.load(MODEL_PTH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)

    logits = _logits_on_array(model, X, device=device, batch_size=8192)
    probs = _stable_sigmoid(logits)

    ts = np.linspace(0.0, 1.0, 101)
    accs, f1s = [], []
    from sklearn.metrics import f1_score
    for t in ts:
        pred = (probs > t).astype(int)
        accs.append((pred == y).mean() * 100.0)
        f1s.append(f1_score(y, pred) * 100.0)

    accs, f1s = np.asarray(accs), np.asarray(f1s)
    best_idx = int(np.argmax(f1s))
    best_t, best_f1 = float(ts[best_idx]), float(f1s[best_idx])

    plt.figure(figsize=(12, 7))
    plt.plot(ts, accs, label="Accuracy (%)", linewidth=2.2)
    plt.plot(ts, f1s,  label="F1 Score (%)", linewidth=2.2)
    plt.axvline(best_t, color="gray", linestyle="--", linewidth=1.2)
    plt.scatter([best_t], [best_f1], s=70)

    # Shift label left/down when near the right edge
    x_off = -0.25 if best_t > 0.90 else 0.03
    y_off = -15 if best_t > 0.90 else -10
    plt.annotate(
        f"Best F1: {best_f1:.1f}% @ {best_t:.2f}",
        xy=(best_t, best_f1),
        xytext=(best_t + x_off, best_f1 + y_off),
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )

    plt.title("KAN - Threshold Optimization")
    plt.xlabel("Threshold"); plt.ylabel("Metric (%)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(THRESH_PNG, dpi=300); plt.close()
    print(f"[OK] Saved threshold sweep figure -> {THRESH_PNG} | Best F1 = {best_f1:.2f}% @ t={best_t:.3f}")

# ------------ SHAP summary (fast) ------------
def run_shap_fast(K=30, explain_n=2000):
    """
    Compute SHAP values for the KAN model, save per-feature importances to CSV (for cross-model Spearman),
    and render the SHAP summary plot. Uses a k-means summarized background for speed.
    """
    if not _HAS_SHAP:
        print("[WARN] shap not installed; skipping SHAP.")
        return

    # --- Load data ---
    X, _ = _load_all_preprocessed()
    feat_dim = X.shape[1]

    # --- Prepare feature names (RAW for CSV/joins; PRETTY only for plots) ---
    # Pull raw names from CSV header if available to ensure consistent ordering.
    feat_names_raw = _load_feature_names(feat_dim)  # exact names for merging across models
    feat_names_pretty = _map_feature_labels(feat_names_raw)  # human-friendly labels for plots

    # --- Summarize background for KernelExplainer (fast & stable) ---
    try:
        background = shap.kmeans(X, K)
    except Exception:
        np.random.seed(42)
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=min(K, X.shape[0]), replace=False)
        background = X[idx]

    # --- Subsample points to explain ---
    rng = np.random.default_rng(42)
    ex_idx = rng.choice(X.shape[0], size=min(explain_n, X.shape[0]), replace=False)
    X_explain = X[ex_idx].astype(np.float32)

    # --- Build a CPU model wrapper for KernelExplainer ---
    device_cpu = torch.device("cpu")
    model = KANModel(input_size=feat_dim)
    state = torch.load(MODEL_PTH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device_cpu)
    model.eval()

    def f_cpu(x_np: np.ndarray) -> np.ndarray:
        """Return probabilities for KernelExplainer on CPU (no gradients)."""
        with torch.no_grad():
            xb = torch.from_numpy(x_np.astype(np.float32))
            logits = model(xb)
            return torch.sigmoid(logits).numpy().ravel()

    # --- Compute SHAP values ---
    try:
        expl = shap.KernelExplainer(f_cpu, background)
        shap_vals = expl.shap_values(X_explain, l1_reg="aic")
        if isinstance(shap_vals, list):  # unify shape if explainer returns a list
            shap_vals = shap_vals[0]
    except Exception as e:
        print(f"[ERROR] KernelExplainer failed: {e}. Skipping SHAP.")
        return

    # --- Aggregate global importance: mean(|SHAP|) per feature ---
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)  # shape: (n_features,)

    # --- Build a DataFrame (RAW + PRETTY + rank) and save to CSV ---
    df_shap = pd.DataFrame({
        "feature": feat_names_raw,            # exact name for cross-model join
        "label_pretty": feat_names_pretty,    # human-readable label for readers
        "mean_abs_shap": mean_abs_shap
    })
    # Rank features by importance (1 = most important); stable ties
    df_shap["rank"] = df_shap["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    df_shap = df_shap.sort_values("rank", kind="stable")
    df_shap.to_csv("KAN_shap_values.csv", index=False)
    print("[OK] Saved CSV with per-feature SHAP importances -> KAN_shap_values.csv")

    # --- Plot SHAP summary using PRETTY labels only for visualization ---
    plt.figure()
    shap.summary_plot(shap_vals, X_explain, feature_names=feat_names_pretty, show=False)
    plt.title("KAN - SHAP summary")
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved SHAP summary figure -> {SHAP_PNG} (fast: K={K}, N={X_explain.shape[0]})")

# ------------ Main ------------
if __name__ == "__main__":
    try:
        run_threshold_all()
    except Exception as e:
        print(f"[WARN] Threshold sweep failed: {e}")
    run_shap_fast(K=30, explain_n=2000)
