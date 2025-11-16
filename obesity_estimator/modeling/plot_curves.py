# obesity_estimator/modeling/plot_curves.py
# -*- coding: utf-8 -*-
"""
obesity_estimator/modeling/plot_curves.py (robusto)

- Lee params.yaml
- Limpia NaN en y y valida estratificación segura para recomponer el mismo split (test)
- Usa models/best_model.joblib
- Genera curvas ROC/PR por clase y macro/micro, y CSVs de resumen

Salidas:
- reports/figures/models/roc_per_class.png
- reports/figures/models/pr_per_class.png
- reports/figures/models/roc_macro_compare.png
- reports/figures/models/pr_macro_compare.png
- reports/roc_auc_by_model.csv
- reports/pr_auc_by_model.csv
"""

from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


def _ensure_dirs():
    Path("reports/figures/models").mkdir(parents=True, exist_ok=True)


def _load_dataset(processed_path: str, target_col: str):
    df = pd.read_csv(processed_path)
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no está en {processed_path}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def _clean_xy(X: pd.DataFrame, y: pd.Series):
    mask = ~y.isna()
    if (~mask).any():
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
    return X, y


def _safe_stratify(y: pd.Series, want_stratify: bool):
    if not want_stratify:
        return None
    vc = y.value_counts(dropna=False)
    if len(vc) < 2 or (vc < 2).any():
        return None
    return y


def _macro_micro_roc(y_test_bin, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test_bin.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    return fpr, tpr, roc_auc, all_fpr, mean_tpr


def _macro_micro_pr(y_test_bin, y_score):
    precision = dict()
    recall = dict()
    ap = dict()
    n_classes = y_test_bin.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    # Micro
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test_bin.ravel(), y_score.ravel()
    )
    ap["micro"] = average_precision_score(y_test_bin, y_score, average="micro")

    # Macro
    ap["macro"] = float(np.mean([ap[i] for i in range(n_classes)]))

    return precision, recall, ap


def main():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    processed_path = cfg["data"]["processed_path"]
    target = cfg["target"]["name"]
    split_cfg = cfg["split"]

    _ensure_dirs()

    # Datos y split test con limpieza + estratificación segura
    X, y = _load_dataset(processed_path, target)
    X, y = _clean_xy(X, y)
    strat = _safe_stratify(y, split_cfg.get("stratify", True))
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=split_cfg.get("test_size", 0.3),
        random_state=split_cfg.get("random_state", 42),
        stratify=strat,
    )

    # Modelo
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError("No se encontró models/best_model.joblib. Ejecuta entrenamiento primero.")
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise AttributeError("El modelo cargado no soporta predict_proba, requerido para curvas ROC/PR.")

    # Probabilidades
    y_score = model.predict_proba(X_test)  # (n_samples, n_classes)

    # Binarización de y
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    class_labels = lb.classes_
    # Caso binario: garantizar 2 columnas
    if y_test_bin.ndim == 1:
        y_test_bin = np.column_stack([1 - y_test_bin, y_test_bin])

    # ROC macro/micro
    fpr, tpr, roc_auc, all_fpr, mean_tpr = _macro_micro_roc(y_test_bin, y_score)

    # PR macro/micro
    precision, recall, ap = _macro_micro_pr(y_test_bin, y_score)

    # ROC por clase
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(class_labels):
        plt.plot(fpr[i], tpr[i], lw=1.2, label=f"Clase {label} (AUC={roc_auc[i]:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curvas ROC por clase")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/figures/models/roc_per_class.png", dpi=150)
    plt.close()

    # PR por clase
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(class_labels):
        plt.plot(recall[i], precision[i], lw=1.2, label=f"Clase {label} (AP={ap[i]:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall por clase")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig("reports/figures/models/pr_per_class.png", dpi=150)
    plt.close()

    # ROC macro vs micro
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-avg ROC (AUC={roc_auc['micro']:.3f})", lw=1.6)
    plt.plot(all_fpr, mean_tpr, label=f"macro-avg ROC (AUC={roc_auc['macro']:.3f})", lw=1.6)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC macro y micro")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/figures/models/roc_macro_compare.png", dpi=150)
    plt.close()

    # PR macro vs micro
    plt.figure(figsize=(8, 6))
    plt.plot(recall["micro"], precision["micro"], label=f"micro-avg PR (AP={ap['micro']:.3f})", lw=1.6)
    # Punto macro (no curva agregada)
    plt.scatter([0.5], [ap["macro"]], label=f"macro-avg AP={ap['macro']:.3f}", s=60)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall macro y micro")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("reports/figures/models/pr_macro_compare.png", dpi=150)
    plt.close()

    # CSVs
    pd.DataFrame(
        {"model": ["best_model"], "roc_auc_macro": [roc_auc["macro"]], "roc_auc_micro": [roc_auc["micro"]]}
    ).to_csv("reports/roc_auc_by_model.csv", index=False)

    pd.DataFrame(
        {"model": ["best_model"], "ap_macro": [ap["macro"]], "ap_micro": [ap["micro"]]}
    ).to_csv("reports/pr_auc_by_model.csv", index=False)

    print("Curvas y métricas agregadas generadas correctamente.")


if __name__ == "__main__":
    main()
