# -*- coding: utf-8 -*-
"""
obesity_estimator/modeling/predict.py (robusto + reportes extra)

- Lee params.yaml
- Limpia NaN en y y valida estratificación segura para recomponer el mismo split
- Carga best_model.joblib
- Guarda:
    - reports/evaluation_results.csv   (métricas agregadas)
    - reports/confusion_matrix.png
    - reports/classification_report_test.csv  (por clase)
    - reports/y_true_pred.csv          (auditoría)
"""

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import seaborn as sns


def _ensure_dirs():
    Path("reports").mkdir(parents=True, exist_ok=True)


def _load_dataset(processed_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
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


def main():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    processed_path = cfg["data"]["processed_path"]
    target = cfg["target"]["name"]
    split_cfg = cfg["split"]

    _ensure_dirs()

    X, y = _load_dataset(processed_path, target)
    X, y = _clean_xy(X, y)
    strat = _safe_stratify(y, split_cfg.get("stratify", True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_cfg.get("test_size", 0.3),
        random_state=split_cfg.get("random_state", 42),
        stratify=strat,
    )

    model = joblib.load("models/best_model.joblib")

    y_pred = model.predict(X_test)

    results = {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    # AUC OVR si hay probabilidades
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                results["roc_auc_ovr"] = float(
                    roc_auc_score(y_test, y_proba, multi_class="ovr")
                )
    except Exception:
        pass

    # 1) métricas agregadas
    pd.DataFrame([results]).to_csv("reports/evaluation_results.csv", index=False)

    # 2) classification report por clase (test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv("reports/classification_report_test.csv", index=True)

    # 3) auditoría y_true/y_pred
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("reports/y_true_pred.csv", index=False)

    # Matriz de confusión
    labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Real")
    plt.xlabel("Predicción")
    plt.title("Matriz de confusión - Test")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    plt.close()

    print("Evaluación finalizada.")
    print("Resultados (test):", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
