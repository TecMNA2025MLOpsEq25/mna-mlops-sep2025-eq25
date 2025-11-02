# obesity_estimator/modeling/plot_curves.py
# -*- coding: utf-8 -*-
"""
Genera curvas ROC (micro/macro y por clase) y PR (micro/macro y por clase)
para cada modelo guardado en models/*.joblib y produce comparativas entre modelos.

Salidas:
- reports/figures/models/<model>/{roc_macro_micro.png, roc_per_class.png, pr_macro_micro.png, pr_per_class.png}
- reports/figures/models/roc_macro_compare.png
- reports/figures/models/pr_macro_compare.png
- reports/roc_auc_by_model.csv
- reports/pr_auc_by_model.csv
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    average_precision_score, precision_recall_curve
)

# -------------------
# Config
# -------------------
REPO = Path(__file__).resolve().parents[2]
DATA_TEST = REPO / "data" / "interim" / "test_prepared.csv"   # contiene X_test + y (col TARGET)
MODELS_DIR = REPO / "models"
FIG_DIR = REPO / "reports" / "figures" / "models"
OUT_ROC = REPO / "reports" / "roc_auc_by_model.csv"
OUT_PR  = REPO / "reports" / "pr_auc_by_model.csv"
TARGET_COL = "NObeyesdad"  # ajusta si tu columna objetivo tiene otro nombre

FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Utilidades
# -------------------
def _ensure_model_dir(name: str) -> Path:
    p = FIG_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def _get_models():
    # Carga todos los *_best.joblib y best_model.joblib si existe
    paths = []
    paths += glob.glob(str(MODELS_DIR / "*_best.joblib"))
    best_path = MODELS_DIR / "best_model.joblib"
    if best_path.exists():
        paths.append(str(best_path))
    return sorted(set(paths))

def _split_X_y(df: pd.DataFrame, target: str):
    assert target in df.columns, f"La columna objetivo '{target}' no está en test_prepared.csv"
    y = df[target].astype(str)
    X = df.drop(columns=[target])
    return X, y

def _prob_or_decision(estimator, X):
    # Devuelve probabilidades (predict_proba) o, si no hay, decision_function normalizada por softmax
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    elif hasattr(estimator, "decision_function"):
        z = estimator.decision_function(X)
        # z shape: (n_samples, n_classes) -> softmax
        z = np.array(z)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    else:
        raise ValueError("El estimador no expone ni predict_proba ni decision_function.")

def _safe_name(p: str) -> str:
    n = Path(p).stem
    if n == "best_model":
        return "best_model"
    return n.replace("_best", "")

# -------------------
# Plot helpers
# -------------------
def plot_roc_curves_one_model(name, y_true_b, probas, classes, outdir: Path):
    # micro
    fpr_micro, tpr_micro, _ = roc_curve(y_true_b.ravel(), probas.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # por clase
    fpr, tpr, roc_auc = {}, {}, {}
    for i, cls in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_b[:, i], probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    roc_auc_macro = auc(all_fpr, mean_tpr)

    # Plot macro/micro
    plt.figure(figsize=(7,6))
    plt.plot(fpr_micro, tpr_micro, lw=2, label=f"micro-avg (AUC={roc_auc_micro:.3f})")
    plt.plot(all_fpr,  mean_tpr,   lw=2, label=f"macro-avg (AUC={roc_auc_macro:.3f})")
    plt.plot([0,1], [0,1], linestyle="--", lw=1, color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {name} (micro/macro)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    (outdir / "roc_macro_micro.png").unlink(missing_ok=True)
    plt.savefig(outdir / "roc_macro_micro.png", dpi=150)
    plt.close()

    # Plot por clase
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{cls} (AUC={roc_auc[i]:.3f})")
    plt.plot([0,1], [0,1], linestyle="--", lw=1, color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC por clase - {name}")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    (outdir / "roc_per_class.png").unlink(missing_ok=True)
    plt.savefig(outdir / "roc_per_class.png", dpi=150)
    plt.close()

    return roc_auc_macro

def plot_pr_curves_one_model(name, y_true_b, probas, classes, outdir: Path):
    # micro
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_b.ravel(), probas.ravel())
    ap_micro = average_precision_score(y_true_b, probas, average="micro")

    # por clase
    prec, rec, ap = {}, {}, {}
    for i, cls in enumerate(classes):
        prec[i], rec[i], _ = precision_recall_curve(y_true_b[:, i], probas[:, i])
        ap[i] = average_precision_score(y_true_b[:, i], probas[:, i])

    # macro AP
    ap_macro = average_precision_score(y_true_b, probas, average="macro")

    # Plot macro/micro
    plt.figure(figsize=(7,6))
    plt.plot(recall_micro, precision_micro, lw=2, label=f"micro-avg (AP={ap_micro:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {name} (micro)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    (outdir / "pr_macro_micro.png").unlink(missing_ok=True)
    plt.savefig(outdir / "pr_macro_micro.png", dpi=150)
    plt.close()

    # Plot por clase
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        plt.plot(rec[i], prec[i], lw=1.5, label=f"{cls} (AP={ap[i]:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall por clase - {name} (macro AP={ap_macro:.3f})")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    (outdir / "pr_per_class.png").unlink(missing_ok=True)
    plt.savefig(outdir / "pr_per_class.png", dpi=150)
    plt.close()

    return ap_macro

def main():
    # 1) Carga test y separa X/y
    df_test = pd.read_csv(DATA_TEST)
    X_test, y_test = _split_X_y(df_test, TARGET_COL)
    classes = sorted(y_test.unique().tolist())
    y_test_b = label_binarize(y_test, classes=classes)

    model_paths = _get_models()
    if not model_paths:
        raise SystemExit("No se encontraron modelos en models/*.joblib")

    roc_summary = []
    pr_summary  = []

    # 2) Curvas por modelo
    macro_roc_all = {}
    macro_pr_all  = {}

    for p in model_paths:
        name = _safe_name(p)
        outdir = _ensure_model_dir(name)

        est = joblib.load(p)
        probas = _prob_or_decision(est, X_test)
        # asegurar forma consistente
        if probas.shape[1] != len(classes):
            raise ValueError(f"{name}: salida proba tiene {probas.shape[1]} clases y se esperaban {len(classes)}.")

        roc_macro = plot_roc_curves_one_model(name, y_test_b, probas, classes, outdir)
        pr_macro  = plot_pr_curves_one_model(name, y_test_b, probas, classes, outdir)

        macro_roc_all[name] = roc_macro
        macro_pr_all[name]  = pr_macro

        roc_summary.append({"model": name, "roc_auc_macro": float(roc_macro)})
        pr_summary.append({"model": name, "ap_macro": float(pr_macro)})

    # 3) Comparativas entre modelos (macro)
    # ROC macro compare
    plt.figure(figsize=(7,6))
    for name, val in sorted(macro_roc_all.items(), key=lambda x: -x[1]):
        plt.plot([0,1], [0,1], alpha=0)  # espacio
        plt.scatter([0.6], [0.1], alpha=0)  # truco layout
        plt.plot([], [], label=f"{name}: AUC={val:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray", lw=1)
    plt.title("Comparativa ROC macro (AUC) por modelo")
    plt.legend(loc="lower right")
    plt.tight_layout()
    (FIG_DIR / "roc_macro_compare.png").unlink(missing_ok=True)
    plt.savefig(FIG_DIR / "roc_macro_compare.png", dpi=150)
    plt.close()

    # PR macro compare
    plt.figure(figsize=(7,6))
    for name, val in sorted(macro_pr_all.items(), key=lambda x: -x[1]):
        plt.plot([], [], label=f"{name}: AP={val:.3f}")
    plt.title("Comparativa PR macro (AP) por modelo")
    plt.legend(loc="lower left")
    plt.tight_layout()
    (FIG_DIR / "pr_macro_compare.png").unlink(missing_ok=True)
    plt.savefig(FIG_DIR / "pr_macro_compare.png", dpi=150)
    plt.close()

    # 4) Guardar CSVs de resumen
    pd.DataFrame(roc_summary).sort_values("roc_auc_macro", ascending=False).to_csv(OUT_ROC, index=False)
    pd.DataFrame(pr_summary).sort_values("ap_macro", ascending=False).to_csv(OUT_PR, index=False)

if __name__ == "__main__":
    main()
