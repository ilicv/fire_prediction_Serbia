# -*- coding: utf-8 -*-
# XGBoost_Train.py — trains on dnn_ready_dataset.csv (already scaled + one-hot)
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, average_precision_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import time


script_start = time.time()


IN_CSV = "dnn_ready_dataset.csv"   # prefer the prepared dataset
MODEL_JSON = "XGBoost_fire_model.json"
LOGLOSS_PNG = "XGBoost_logloss_plot.png"
IMP_PNG = "XGBoost_feature_importance_gain.png"
CM_PREFIX = "XGBoost"
REPORT_TXT = "XGBoost_evaluation_report.txt"

def _fmt_seconds(s: float) -> str:
    s = int(max(0, s)); h, r = divmod(s,3600); m, s = divmod(r,60)
    return f"{h:d}:{m:02d}:{s:02d}"


def pick_tree_method():
    """Try to use GPU if available, else CPU."""
    # Many xgboost builds include GPU; if not, fallback is safe
    try:
        params = dict(tree_method="gpu_hist", predictor="gpu_predictor")
        # quick dry run to confirm support (no training)
        _ = xgb.get_config()
        return params
    except Exception:
        return dict(tree_method="hist", predictor="auto")

def find_best_threshold(probs, y_true, metric="f1"):
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best = 0.5, -1.0
    for t in ts:
        pred = (probs > t).astype(int)
        score = f1_score(y_true, pred) if metric == "f1" else average_precision_score(y_true, probs)
        if score > best:
            best, best_t = score, t
    return float(best_t), float(best)

def evaluate(y_true, y_pred, title):
    print(f"\n?? {title} Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    print("    Confusion Matrix Values:")
    print(f"    TN: {tn} ({tn/total:.1%})  FP: {fp} ({fp/total:.1%})")
    print(f"    FN: {fn} ({fn/total:.1%})  TP: {tp} ({tp/total:.1%})")

    # Annotated confusion matrix
    labels = np.array([
        [f"{tn}\n({tn/total:.1%})", f"{fp}\n({fp/total:.1%})"],
        [f"{fn}\n({fn/total:.1%})", f"{tp}\n({tp/total:.1%})"]
    ])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
                xticklabels=["No Fire","Fire"],
                yticklabels=["No Fire","Fire"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"{title} - Confusion Matrix")
    plt.tight_layout()
    out_png = f"{CM_PREFIX}_{title.lower()}_confusion_matrix.png"
    plt.savefig(out_png, dpi=300); plt.show()
    return out_png

def main():
    if not os.path.exists(IN_CSV):
        # Fallback to older extractor output if needed
        alt = "final_dataset_from_resampled.csv"
        if os.path.exists(alt):
            print(f"?? {IN_CSV} not found, using {alt} instead.")
            csv_path = alt
        else:
            raise FileNotFoundError(f"CSV not found: {IN_CSV}")
    else:
        csv_path = IN_CSV

    print("?? Loading dataset...")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Ensure binary label
    df["label"] = (df["label"].astype(float) > 0).astype(int)

    # Features: use all columns except label (coords are not present in prepared CSV)
    y = df["label"].values.astype(np.int32)
    X = df.drop(columns=["label"])
    feature_names = X.columns.tolist()

    print("?? Stratified split train/val/test = 80/10/10 ...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"?? Train: {len(X_train):,}  | Val: {len(X_val):,}  | Test: {len(X_test):,}")

    # scale_pos_weight from training split only
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale = (neg / max(1, pos))
    print(f"?? scale_pos_weight = {scale:.3f}  (neg={neg:,}, pos={pos:,})")

    # Choose tree method (GPU if possible)
    tm = pick_tree_method()

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "seed": 42,
        **tm
    }

    print("?? Training with early stopping (up to 4000 rounds)...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feature_names)

    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=4000,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=True
    )

    print("?? Saving model...")
    model.save_model(MODEL_JSON)

    # Plot training/validation logloss
    train_logloss = evals_result["train"]["logloss"]
    val_logloss   = evals_result["validation"]["logloss"]
    plt.figure(figsize=(7,5))
    plt.plot(train_logloss, label="Train Logloss")
    plt.plot(val_logloss,   label="Validation Logloss")
    plt.axvline(model.best_iteration, color="red", linestyle="--", label="Best Iter")
    plt.title("Training vs Validation Logloss")
    plt.xlabel("Boosting Round"); plt.ylabel("Logloss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(LOGLOSS_PNG, dpi=300); plt.show()

    # === Threshold tuning on validation (maximize F1) ===
    best_iter = model.best_iteration
    val_probs = model.predict(dval, iteration_range=(0, best_iter+1))
    best_t, best_f1 = find_best_threshold(val_probs, y_val, metric="f1")
    print(f"?? Best threshold on validation = {best_t:.2f} (F1={best_f1:.3f})")

    # Predictions
    def predict_with_t(dmat, t):
        return (model.predict(dmat, iteration_range=(0, best_iter+1)) > t).astype(int)

    y_train_pred = predict_with_t(dtrain, best_t)
    y_val_pred   = predict_with_t(dval,   best_t)
    y_test_prob  = model.predict(dtest, iteration_range=(0, best_iter+1))
    y_test_pred  = (y_test_prob > best_t).astype(int)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc   = accuracy_score(y_val,   y_val_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)
    pr_auc    = average_precision_score(y_test, y_test_prob)
    roc_auc   = roc_auc_score(y_test, y_test_prob)

    print(f"\n? Final metrics (Test): Acc {test_acc:.4f} | PR-AUC {pr_auc:.4f} | ROC-AUC {roc_auc:.4f}")

    # Evaluate & plot confusion matrices
    evaluate(y_train, y_train_pred, "Training")
    evaluate(y_val,   y_val_pred,   "Validation")
    evaluate(y_test,  y_test_pred,  "Test")

    # Feature importance (by gain)
    print("?? Plotting feature importance (by gain)...")
    ax = xgb.plot_importance(
        model, importance_type="gain", max_num_features=20,
        height=0.5, xlabel="Average Gain",
        title="Top 20 Features (by Gain)", show_values=False, grid=True
    )
    new_labels = [label.get_text().replace("_resampled", "") for label in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels)
    plt.tight_layout(); plt.savefig(IMP_PNG, dpi=300); plt.show()

    # Save test split for explanation tools
    np.save("xgb_X_test.npy", X_test.values)
    np.save("xgb_y_test.npy", y_test)

    # Save text report
    report = classification_report(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()

    total_runtime = time.time() - script_start
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"\nTotal runtime: {_fmt_seconds(total_runtime)}\n")
        f.write(f"Best iteration: {best_iter}\n")
        f.write(f"Best val threshold: {best_t:.2f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Test ROC-AUC: {roc_auc:.4f}\n\n")
        f.write("Test Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix Values:\n")
        f.write(f"  TN: {tn} ({tn/total:.1%})  FP: {fp} ({fp/total:.1%})\n")
        f.write(f"  FN: {fn} ({fn/total:.1%})  TP: {tp} ({tp/total:.1%})\n")

    print(f"Total runtime: {_fmt_seconds(total_runtime)}")
    print("?? Saved:", REPORT_TXT)
    print("?? Saved model:", MODEL_JSON)

if __name__ == "__main__":
    main()
