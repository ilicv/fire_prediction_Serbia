# cross_model_spearman.py
# -*- coding: utf-8 -*-
"""
End-to-end concordance processing for SHAP feature importances across models.
Inputs (expected in the working directory):
  - DNN_shap_values.csv    (feature,label_pretty,mean_abs_shap,rank)
  - KAN_shap_values.csv    (feature,label_pretty,mean_abs_shap,rank)
  - XGB_shap_values.csv    (feature,label_pretty,mean_abs_shap,rank)

Outputs:
  - Cross_model_Spearman.csv          -> ρ by mean|SHAP| and by rank
  - Feature_Importances_Merged.csv    -> merged table (values + ranks)
  - Rank_Divergences.csv              -> features with largest rank disagreement
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# -------- Configuration --------
DNN_CSV = "DNN_shap_values.csv"
KAN_CSV = "KAN_shap_values.csv"
XGB_CSV = "XGB_shap_values.csv"

OUT_SPEARMAN = "Cross_model_Spearman.csv"
OUT_MERGED = "Feature_Importances_Merged.csv"
OUT_DIVERGENCES = "Rank_Divergences.csv"

# How many top disagreements to save (set None to keep all)
TOP_K_DIVERGENCES = 50


def _load_model_csv(path: str, tag: str) -> pd.DataFrame:
    """Load one model's SHAP CSV and rename columns with model tag."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)

    # Validate required columns
    required = {"feature", "mean_abs_shap", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    # Keep only what we need + pretty label for optional reporting
    keep = ["feature", "label_pretty", "mean_abs_shap", "rank"]
    for c in keep:
        if c not in df.columns:
            # label_pretty is optional; create a passthrough if missing
            if c == "label_pretty":
                df[c] = df["feature"]
            else:
                raise ValueError(f"{path} missing required column '{c}'")

    df = df[keep].copy()
    df = df.rename(
        columns={
            "label_pretty": f"label_pretty_{tag}",
            "mean_abs_shap": f"mean_abs_shap_{tag}",
            "rank": f"rank_{tag}",
        }
    )
    return df


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rho with NaN-safe policy."""
    rho = spearmanr(a, b, nan_policy="omit").correlation
    # Handle rare all-constant edge case where rho can be nan
    return float(0.0 if np.isnan(rho) else rho)


def main():
    # ---- Load inputs ----
    dnn = _load_model_csv(DNN_CSV, "dnn")
    kan = _load_model_csv(KAN_CSV, "kan")
    xgb = _load_model_csv(XGB_CSV, "xgb")

    # ---- Inner join on 'feature' to ensure common universe ----
    merged = dnn.merge(kan, on="feature").merge(xgb, on="feature")

    if merged.empty:
        raise RuntimeError("No overlapping features across the three CSVs.")

    # ---- Compute Spearman on mean|SHAP| and on explicit ranks ----
    rho_vals = {
        "DNN–KAN": _spearman(merged["mean_abs_shap_dnn"], merged["mean_abs_shap_kan"]),
        "DNN–XGB": _spearman(merged["mean_abs_shap_dnn"], merged["mean_abs_shap_xgb"]),
        "KAN–XGB": _spearman(merged["mean_abs_shap_kan"], merged["mean_abs_shap_xgb"]),
    }
    rho_ranks = {
        "DNN–KAN": _spearman(merged["rank_dnn"], merged["rank_kan"]),
        "DNN–XGB": _spearman(merged["rank_dnn"], merged["rank_xgb"]),
        "KAN–XGB": _spearman(merged["rank_kan"], merged["rank_xgb"]),
    }

    summary = pd.DataFrame(
        {
            "Model pair": ["DNN–KAN", "DNN–XGB", "KAN–XGB"],
            "Spearman ρ (by mean |SHAP|)": [
                rho_vals["DNN–KAN"],
                rho_vals["DNN–XGB"],
                rho_vals["KAN–XGB"],
            ],
            "Spearman ρ (by rank)": [
                rho_ranks["DNN–KAN"],
                rho_ranks["DNN–XGB"],
                rho_ranks["KAN–XGB"],
            ],
        }
    ).round(3)

    # ---- Save Spearman summary ----
    summary.to_csv(OUT_SPEARMAN, index=False)

    # ---- Build a comprehensive merged table and save ----
    # Keep a single human-readable label (prefer DNN pretty if available)
    merged["label_pretty"] = merged["label_pretty_dnn"]
    cols = [
        "feature",
        "label_pretty",
        "mean_abs_shap_dnn",
        "mean_abs_shap_kan",
        "mean_abs_shap_xgb",
        "rank_dnn",
        "rank_kan",
        "rank_xgb",
    ]
    merged_out = merged[cols].copy()

    # Add average rank and rank dispersion metrics
    merged_out["rank_mean"] = merged_out[["rank_dnn", "rank_kan", "rank_xgb"]].mean(axis=1)
    merged_out["rank_std"] = merged_out[["rank_dnn", "rank_kan", "rank_xgb"]].std(axis=1, ddof=0)
    merged_out["rank_max_diff"] = merged_out[
        ["rank_dnn", "rank_kan", "rank_xgb"]
    ].max(axis=1) - merged_out[["rank_dnn", "rank_kan", "rank_xgb"]].min(axis=1)

    # Sort by average rank (most important first)
    merged_out = merged_out.sort_values(["rank_mean", "rank_std"], ascending=[True, True])
    merged_out.to_csv(OUT_MERGED, index=False)

    # ---- Save top rank divergences (largest disagreement) ----
    divergences = merged_out.sort_values("rank_max_diff", ascending=False)
    if TOP_K_DIVERGENCES is not None:
        divergences = divergences.head(TOP_K_DIVERGENCES)
    divergences.to_csv(OUT_DIVERGENCES, index=False)

    # ---- Console summary ----
    print(f"[OK] Saved: {OUT_SPEARMAN}")
    print(f"[OK] Saved: {OUT_MERGED}")
    print(f"[OK] Saved: {OUT_DIVERGENCES}")
    print("\nSpearman summary:\n", summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
