import os, json
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
import matplotlib.pyplot as plt

# Paths (DVC ya los declara como deps/outs)
TEST_PATH = "data/interim/test_prepared.csv"
MODEL_PATH = "models/best_model.pkl"
REPORTS_DIR = "reports"
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")
CONF_MAT_PATH = os.path.join(REPORTS_DIR, "confusion_matrix_eval.png")

os.makedirs(REPORTS_DIR, exist_ok=True)

def load_data_and_model():
    df = pd.read_csv(TEST_PATH)
    # Infieren target: usa 'NObeyesdad' si existe; si no, usa la última columna
    target_col = "NObeyesdad" if "NObeyesdad" in df.columns else df.columns[-1]
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    model = joblib.load(MODEL_PATH)
    return X_test, y_test, model, target_col

def maybe_proba(model, X):
    # intenta probabilities si existen; si no, regresa None
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None

def main():
    X_test, y_test, model, target_col = load_data_and_model()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(CONF_MAT_PATH)
    plt.close()

    # ROC-AUC (solo si hay predict_proba y es multiclase)
    roc_auc_ovr = None
    proba = maybe_proba(model, X_test)
    if proba is not None:
        try:
            # Binariza internamente y calcula macro-ovr
            roc_auc_ovr = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        except Exception:
            roc_auc_ovr = None

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "roc_auc_ovr": float(roc_auc_ovr) if roc_auc_ovr is not None else None,
        "n_samples": int(len(y_test)),
        "target": target_col
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[evaluate] metrics.json ->", METRICS_PATH)
    print("[evaluate] confusion_matrix_eval.png ->", CONF_MAT_PATH)

if __name__ == "__main__":
    main()