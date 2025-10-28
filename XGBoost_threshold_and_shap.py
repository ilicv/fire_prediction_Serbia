# -*- coding: utf-8 -*-
"""
XGBoost_threshold_and_shap.py
-------------------------------------
Post-hoc analysis for the trained XGBoost wildfire model.

Outputs:
- XGB_threshold_optimization.png
- XGB_shap_summary.png

Expected artifacts:
- Model:  XGBoost_fire_model.json
- Arrays: xgb_X_test.npy, xgb_y_test.npy (optional; falls back to CSV)
- CSV:    dnn_ready_dataset.csv (for feature names)
"""

import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

IN_CSV     = "dnn_ready_dataset.csv"
MODEL_JSON = "XGBoost_fire_model.json"
X_TEST_NPY = "xgb_X_test.npy"
Y_TEST_NPY = "xgb_y_test.npy"
THRESH_PNG = "XGB_threshold_optimization.png"
SHAP_PNG   = "XGB_shap_summary.png"

# ==== Label mapping for SHAP ====
def _map_feature_labels(feat_names):
    """Human-friendly labels for Aspect / Land use codes."""
    aspect_map = {
        "1": "North","2": "North-East","3": "East","4": "South-East","5": "South",
        "6": "South-West","7": "West","8": "North-West","9": "Unexposed",
    }
    landuse_map = {
        "1": "Water bodies","2": "Forests","4": "Seasonally flooded areas",
        "5": "Agricultural land","7": "Settlements","8": "Barren surfaces",
        "9": "Snow / Ice","10": "Clouds (no data)","11": "Pastures / Grasslands",
    }
    def _rename(name):
        n = name.replace(" ", "")
        m = re.match(r"(?i)Aspect(?:==|=)(\d+)", n)
        if m: return f"Aspect: {aspect_map.get(m.group(1), m.group(1))}"
        m = re.match(r"(?i)Land[_\s]*use(?:==|=)(\d+)", n)
        if m: return f"Land use: {landuse_map.get(m.group(1), m.group(1))}"
        return name
    return [_rename(n) for n in feat_names]

# ==== Data / model helpers ====
def _load_model():
    if not os.path.exists(MODEL_JSON):
        raise FileNotFoundError(f"Model not found: {MODEL_JSON}")
    booster = xgb.Booster()
    booster.load_model(MODEL_JSON)
    return booster

def _load_data_for_eval():
    """Prefer xgb_* npy arrays; fall back to CSV. Return (X, y, feature_names)."""
    feat_names = None
    if os.path.exists(IN_CSV):
        try:
            dfh = pd.read_csv(IN_CSV, nrows=5)
            feat_names = [c for c in dfh.columns if c != "label"] if "label" in dfh.columns else list(dfh.columns)
        except Exception:
            feat_names = None

    if os.path.exists(X_TEST_NPY) and os.path.exists(Y_TEST_NPY):
        X = np.load(X_TEST_NPY)
        y = np.load(Y_TEST_NPY).astype(int).ravel()
        if (feat_names is None) or (len(feat_names) != X.shape[1]):
            feat_names = [f"feat_{i}" for i in range(X.shape[1])]
        return X, y, feat_names

    if not os.path.exists(IN_CSV):
        raise FileNotFoundError("Neither xgb_* arrays nor CSV found.")
    df = pd.read_csv(IN_CSV)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    y = df["label"].to_numpy().astype(int).ravel()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    if (feat_names is None) or (len(feat_names) != X.shape[1]):
        feat_names = [f"feat_{i}" for i in range(X.shape[1])]
    return X, y, feat_names

def _predict_proba(booster, X_np, feature_names=None, best_ntree_limit=None):
    """XGBoost probability predictions."""
    dmat = xgb.DMatrix(X_np, feature_names=feature_names)
    if best_ntree_limit is not None:
        return booster.predict(dmat, iteration_range=(0, int(best_ntree_limit)))
    return booster.predict(dmat)

# ==== Threshold sweep ====
def run_threshold_sweep():
    booster = _load_model()
    X, y, feat_names = _load_data_for_eval()

    best_iter = None
    try:
        best_iter_attr = booster.attr("best_iteration")
        if best_iter_attr is not None:
            best_iter = int(best_iter_attr) + 1
    except Exception:
        best_iter = None

    probs = _predict_proba(booster, X, feature_names=feat_names, best_ntree_limit=best_iter)

    ts = np.linspace(0.0, 1.0, 101)
    accs, f1s = [], []
    for t in ts:
        pred = (probs > t).astype(int)
        accs.append((pred == y).mean() * 100.0)
        f1s.append(f1_score(y, pred) * 100.0)

    accs = np.asarray(accs); f1s = np.asarray(f1s)
    best_idx = int(np.argmax(f1s))
    best_t, best_f1 = float(ts[best_idx]), float(f1s[best_idx])

    plt.figure(figsize=(12, 7))
    plt.plot(ts, accs, label="Accuracy (%)", linewidth=2.2)
    plt.plot(ts, f1s,  label="F1 Score (%)", linewidth=2.2)
    plt.axvline(best_t, color="gray", linestyle="--", linewidth=1.2)
    plt.scatter([best_t], [best_f1], s=70)

    x_off = -0.25 if best_t > 0.90 else 0.03
    y_off = -15 if best_t > 0.90 else -10
    plt.annotate(
        f"Best F1: {best_f1:.1f}% @ {best_t:.2f}",
        xy=(best_t, best_f1),
        xytext=(best_t + x_off, best_f1 + y_off),
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )

    plt.title("XGBoost - Threshold Optimization")
    plt.xlabel("Threshold"); plt.ylabel("Metric (%)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(THRESH_PNG, dpi=300); plt.close()
    print(f"[OK] Saved threshold sweep figure -> {THRESH_PNG} | Best F1 = {best_f1:.2f}% @ t={best_t:.3f}")

# ==== SHAP summary (fast & robust) ====
def run_shap_summary(K=30, explain_n=2000):
    """
    Compute SHAP values for XGBoost, save per-feature importances to CSV (for cross-model Spearman),
    and render the SHAP summary plot. Uses k-means summarized background for speed.
    """
    try:
        import shap
    except Exception:
        print("[WARN] shap not installed; skipping SHAP.")
        return

    # --- Load model and data ---
    print("[INFO] Loading model/data for SHAP...")
    booster = _load_model()
    X, _, feat_names_from_loader = _load_data_for_eval()
    feat_dim = X.shape[1]

    # --- Summarize background with k-means (fast & stable) ---
    print("[INFO] Summarizing background with k-means...")
    try:
        background = shap.kmeans(X, K)  # shape: (K, d)
    except Exception:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=min(K, X.shape[0]), replace=False)
        background = X[idx]

    # --- Subsample points to explain ---
    print("[INFO] Sampling points to explain...")
    rng = np.random.default_rng(42)
    ex_idx = rng.choice(X.shape[0], size=min(explain_n, X.shape[0]), replace=False)
    X_explain = X[ex_idx].astype(np.float32)

    # --- Build SHAP explainer for XGBoost ---
    # Prefer modern API; fall back to legacy TreeExplainer if needed.
    try:
        print("[INFO] Building SHAP Explainer (tree)...")
        masker = shap.maskers.Independent(background)  # robust tabular masker
        explainer = shap.Explainer(booster, masker, algorithm="tree", model_output="raw")
        print("[INFO] Computing SHAP values...")
        shap_values = explainer(X_explain, check_additivity=False).values
    except Exception as e_primary:
        print(f"[WARN] Explainer(tree) failed: {e_primary}. Falling back to legacy TreeExplainer.")
        try:
            explainer = shap.TreeExplainer(booster, model_output="raw")
            shap_values = explainer.shap_values(X_explain, check_additivity=False)
        except Exception as e_legacy:
            print(f"[ERROR] SHAP fallback failed: {e_legacy}")
            return

    # --- Prepare feature names (raw + pretty) ---
    # Keep RAW names for CSV merging across models; use PRETTY only for plotting.
    if os.path.exists(IN_CSV):
        try:
            head = pd.read_csv(IN_CSV, nrows=5)
            feat_names_raw = [c for c in head.columns if c != "label"] if "label" in head.columns else list(head.columns)
            if len(feat_names_raw) != feat_dim:
                feat_names_raw = [f"feat_{i}" for i in range(feat_dim)]
        except Exception:
            feat_names_raw = [f"feat_{i}" for i in range(feat_dim)]
    else:
        feat_names_raw = feat_names_from_loader if len(feat_names_from_loader) == feat_dim else [f"feat_{i}" for i in range(feat_dim)]

    feat_names_pretty = _map_feature_labels(feat_names_raw)

    # --- Compute mean absolute SHAP per feature (global importance proxy) ---
    shap_values_np = np.array(shap_values)
    mean_abs_shap = np.abs(shap_values_np).mean(axis=0)  # (n_features,)

    # --- Build DataFrame and assign ranks (1 = most important) ---
    df_shap = pd.DataFrame({
        "feature": feat_names_raw,           # raw key for cross-model join
        "label_pretty": feat_names_pretty,  # human-readable label
        "mean_abs_shap": mean_abs_shap
    })
    df_shap["rank"] = df_shap["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    df_shap = df_shap.sort_values("rank", kind="stable")

    # --- Save CSV (for cross-model Spearman correlation) ---
    df_shap.to_csv("XGB_shap_values.csv", index=False)
    print("[OK] Saved CSV with per-feature SHAP importances -> XGB_shap_values.csv")

    # --- Render SHAP summary plot (use pretty labels only for plotting) ---
    print("[INFO] Rendering SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values_np, X_explain, feature_names=feat_names_pretty, show=False)
    plt.title("XGBoost - SHAP summary")
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved SHAP summary figure -> {SHAP_PNG} (K={K}, N={X_explain.shape[0]})")

# ==== main ====
if __name__ == "__main__":
    run_threshold_sweep()
    run_shap_summary(K=30, explain_n=2000)
