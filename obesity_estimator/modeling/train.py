# -*- coding: utf-8 -*-
"""
obesity_estimator/modeling/train.py (robusto)

- Lee config de params.yaml
- Limpia NaN en y (target) y sincroniza X
- Valida estratificación: si alguna clase queda con <2 muestras o hay problema, desactiva stratify
- Guarda metrics.json y classification_report.csv
- Guarda best_model.joblib
"""

import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from obesity_estimator.pipeline import build_pipeline


def _ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)


def _load_dataset(processed_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(processed_path)
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no está en {processed_path}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def _clean_xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Elimina filas con y NaN y sincroniza X; registra conteos previos y posteriores."""
    info = {}
    info["rows_before"] = int(len(y))
    mask = ~y.isna()
    dropped = int((~mask).sum())
    if dropped > 0:
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
    info["rows_after_drop_target_nan"] = int(len(y))
    info["dropped_target_nan"] = dropped
    return X, y, info


def _safe_stratify(y: pd.Series, want_stratify: bool) -> Tuple[bool, dict]:
    """Verifica si estratificar es posible: >=2 muestras por clase y >=2 clases."""
    info = {"requested": bool(want_stratify)}
    if not want_stratify:
        info["used"] = False
        info["reason"] = "disabled_by_config"
        return False, info

    vc = y.value_counts(dropna=False)
    info["class_counts"] = vc.to_dict()
    if len(vc) < 2:
        info["used"] = False
        info["reason"] = "only_one_class"
        return False, info
    if (vc < 2).any():
        info["used"] = False
        info["reason"] = "some_class_lt2"
        return False, info

    info["used"] = True
    info["reason"] = "ok"
    return True, info


def main():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    features_cfg = cfg["features"]
    target_cfg = cfg["target"]
    model_cfg = cfg["model"]

    processed_path = data_cfg["processed_path"]
    target = target_cfg["name"]

    _ensure_dirs()

    # Carga de datos
    X, y = _load_dataset(processed_path, target)

    # Limpieza de NaN en y
    X, y, clean_info = _clean_xy(X, y)

    # Estratificación segura
    use_stratify, strat_info = _safe_stratify(y, split_cfg.get("stratify", True))
    stratify_vec = y if use_stratify else None

    # División train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=split_cfg.get("test_size", 0.3),
        random_state=split_cfg.get("random_state", 42),
        stratify=stratify_vec,
    )

    # Pipeline y entrenamiento
    pipe = build_pipeline(features_cfg, model_cfg)
    pipe.fit(X_train, y_train)

    # Evaluación
    y_pred = pipe.predict(X_val)
    metrics = {
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision_macro": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
    }

    # AUC OVR si aplica
    try:
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_val)
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_val, y_proba, multi_class="ovr")
                )
    except Exception:
        pass

    # Reporte y logs de calidad
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv("reports/classification_report.csv", index=True)

    quality_log = {
        "clean_info": clean_info,
        "stratify_info": strat_info,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }
    with open("reports/data_quality.json", "w") as f:
        json.dump(quality_log, f, indent=2)

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, "models/best_model.joblib")

    print("Entrenamiento finalizado.")
    print("Métricas (validación):", json.dumps(metrics, indent=2))
    print("Calidad de datos:", json.dumps(quality_log, indent=2))


if __name__ == "__main__":
    main()
