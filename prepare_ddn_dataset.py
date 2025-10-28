# prepare_ddn_dataset.py  (for extractor outputs with pre-made one-hot)
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---- Inputs / outputs ----
CANDIDATE_INPUTS = [
    "final_dataset_from_resampled_sampled.csv",  # prefer balanced/sampled
    "final_dataset_from_resampled.csv",          # fallback (full)
]
OUTPUT_ALL = "dnn_ready_dataset.csv"
OUTPUT_MINMAX = "min_max_values.csv"
OUTPUT_FEATURES = "feature_columns.csv"
OUTPUT_META = "prep_meta.json"  # small helper (what was scaled, counts, etc.)

# ---- Load ----
in_path = None
for p in CANDIDATE_INPUTS:
    if os.path.exists(p):
        in_path = p
        break
if in_path is None:
    raise FileNotFoundError(f"None of these CSVs were found: {CANDIDATE_INPUTS}")

print(f"?? Loading dataset: {in_path}")
df = pd.read_csv(in_path)

# ---- Label: ensure binary 0/1 (extractor already does this; keep idempotent) ----
if "label" not in df.columns:
    raise ValueError("Expected 'label' column in the CSV.")
df["label"] = (df["label"].astype(float) > 0).astype(int)

# ------------------------------------------------------
# Optional: enforce negative:positive ratio
# Example:
#   1.0 -> equal numbers of negatives and positives (50/50)
#   1.5 -> 1.5x more negatives than positives
#   2.0 -> 2x more negatives than positives
# ------------------------------------------------------
TARGET_NEG_POS_RATIO = 1.5   # adjust here

pos_idx = df.index[df["label"] == 1]
neg_idx = df.index[df["label"] == 0]

if len(pos_idx) > 0 and len(neg_idx) > 0:
    target_neg = int(min(len(neg_idx), TARGET_NEG_POS_RATIO * len(pos_idx)))
    rng = np.random.RandomState(42)
    neg_keep = rng.choice(neg_idx, size=target_neg, replace=False)
    keep_idx = np.concatenate([pos_idx, neg_keep])
    df = df.loc[keep_idx].sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"? Enforced ratio {TARGET_NEG_POS_RATIO}:1 "
          f"(pos={len(pos_idx):,}, neg_kept={target_neg:,}, total={len(df):,})")

# ---- Drop NaNs ----
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"?? Removed {before - len(df)} rows with NaNs; kept {len(df)} rows.")

# ---- Split features vs label; drop coordinates ----
drop_cols = [c for c in ["X", "Y", "label"] if c in df.columns]
X = df.drop(columns=drop_cols).copy()
y = df["label"].copy()

# ---- Identify one-hot vs numeric ----
# One-hot columns are already created by the extractor and contain '==' in name.
ohe_cols = [c for c in X.columns if "==" in c]
numeric_cols = [c for c in X.columns if c not in ohe_cols]

print("?? One-hot columns (kept as 0/1):", len(ohe_cols))
print("?? Numeric columns to scale    :", len(numeric_cols))

# ---- Scale numeric only; keep OHE as-is ----
if numeric_cols:
    scaler = MinMaxScaler()
    X_num = pd.DataFrame(scaler.fit_transform(X[numeric_cols]),
                         columns=numeric_cols, index=X.index)
else:
    scaler = None
    X_num = pd.DataFrame(index=X.index, columns=[])

X_ohe = X[ohe_cols].copy() if ohe_cols else pd.DataFrame(index=X.index, columns=[])

# ---- Combine back (keep a stable order: numeric then one-hot) ----
X_final = pd.concat([X_num, X_ohe], axis=1)
X_final["label"] = y.values

# Optional: reduce float size to save disk
for c in X_num.columns:
    X_final[c] = X_final[c].astype("float32")

# ---- Save outputs ----
X_final.to_csv(OUTPUT_ALL, index=False)
print(f"? Saved normalized dataset ? {OUTPUT_ALL}")

# Save min/max for numeric only
if scaler is not None and numeric_cols:
    min_max_df = pd.DataFrame({
        "feature": numeric_cols,
        "min": scaler.data_min_,
        "max": scaler.data_max_
    })
else:
    min_max_df = pd.DataFrame(columns=["feature", "min", "max"])
min_max_df.to_csv(OUTPUT_MINMAX, index=False)
print(f"?? Saved numeric min/max ? {OUTPUT_MINMAX}")

# Save feature order (for inference time)
pd.Series(list(X_num.columns) + list(X_ohe.columns), name="feature").to_csv(OUTPUT_FEATURES, index=False)
print(f"?? Saved feature order ? {OUTPUT_FEATURES}")

# Small meta file (handy for downstream scripts)
meta = {
    "input_csv": in_path,
    "n_rows": int(len(df)),
    "n_features_total": int(X_final.shape[1]-1),
    "n_numeric_scaled": int(len(numeric_cols)),
    "n_ohe": int(len(ohe_cols)),
}
with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"?? Wrote meta ? {OUTPUT_META}")

# Class balance report
pos = int((y == 1).sum())
neg = int((y == 0).sum())
print(f"?? Class balance ? positives: {pos:,}, negatives: {neg:,}  ({pos/(pos+neg+1e-9):.2%} positive)")
