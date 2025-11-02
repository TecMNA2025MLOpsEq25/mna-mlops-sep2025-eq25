# obesity_estimator/modeling/train.py
# -*- coding: utf-8 -*-
"""
Entrenamiento con opción de búsqueda de hiperparámetros (Grid/Randomized).
- Lee params.yaml
- Construye Pipeline (preprocesamiento + estimador)
- Si model.search.enabled: corre búsqueda y serializa mejores resultados
- Guarda:
  - reports/metrics.json (validación)
  - reports/hpo/best_params.json
  - reports/hpo/rankings.csv
  - reports/hpo/cv_results.csv
  - models/best_model.joblib
"""

from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from obesity_estimator.pipeline import build_pipeline
from obesity_estimator.modeling.search import (
    build_estimator, build_search_space, select_searcher
)


def _ensure_dirs():
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/hpo").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)


def _load_dataset(processed_path: str, target_col: str):
    df = pd.read_csv(processed_path)
    # limpieza mínima del target
    df = df[~df[target_col].isna()].reset_index(drop=True)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def _safe_stratify(y, want=True):
    if not want:
        return None
    vc = y.value_counts(dropna=False)
    if len(vc) < 2 or (vc < 2).any():
        return None
    return y


def _evaluate(model, X, y, labels=None):
    y_pred = model.predict(X)
    metrics = {
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0))
    }
    # AUC OVR si hay probas
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y, y_proba, multi_class="ovr"))
    except Exception:
        pass
    return metrics


def main():
    _ensure_dirs()

    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    target_cfg = cfg["target"]
    features_cfg = cfg["features"]
    split_cfg = cfg["split"]
    model_cfg = cfg["model"]

    X, y = _load_dataset(data_cfg["processed_path"], target_cfg["name"])
    strat = _safe_stratify(y, split_cfg.get("stratify", True))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=split_cfg.get("test_size", 0.3),
        random_state=split_cfg.get("random_state", 42),
        stratify=strat,
    )

    # Construye pipeline: preprocesador + estimador base
    base_estimator = build_estimator(model_cfg)
    pipe = build_pipeline(features_cfg, model_cfg, estimator=base_estimator)

    # ¿Activamos búsqueda?
    search_cfg = (model_cfg.get("search") or {})
    if search_cfg.get("enabled", False):
        # Espacio de hiperparámetros
        space = build_search_space(model_cfg)
        # Prefija con el nombre del paso del estimador dentro del pipeline
        # asumiendo que el estimador está bajo el nombre 'clf'
        param_grid = {f"clf__{k}": v for (k, v) in space.items()}

        Searcher = select_searcher(search_cfg.get("kind", "grid"))
        common_kwargs = dict(
            scoring=target_cfg.get("metric_primary", "f1_macro"),
            n_jobs=search_cfg.get("n_jobs", -1),
            cv=search_cfg.get("cv", 5),
            verbose=search_cfg.get("verbose", 1),
            refit=True,  # reentrena el mejor sobre todo el train
        )

        if search_cfg.get("kind", "grid") == "random":
            n_iter = search_cfg.get("n_iter", 30)
            search = Searcher(pipe, param_distributions=param_grid, n_iter=n_iter, **common_kwargs)
        else:
            search = Searcher(pipe, param_grid=param_grid, **common_kwargs)

        search.fit(X_train, y_train)

        # Serializa resultados HPO
        best_params = search.best_params_
        # Limpia prefijo clf__
        best_params_clean = {k.replace("clf__", ""): v for k, v in best_params.items()}

        pd.DataFrame(search.cv_results_).to_csv("reports/hpo/cv_results.csv", index=False)

        # Ranking
        cols_rank = ["rank_test_score", "mean_test_score", "std_test_score", "params"]
        pd.DataFrame(search.cv_results_)[cols_rank].sort_values("rank_test_score").to_csv(
            "reports/hpo/rankings.csv", index=False
        )

        with open("reports/hpo/best_params.json", "w") as f:
            json.dump(best_params_clean, f, indent=2)

        best_model = search.best_estimator_
    else:
        # Entrena sin búsqueda
        best_model = pipe.fit(X_train, y_train)

    # Evalúa en validación
    metrics_val = _evaluate(best_model, X_val, y_val)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics_val, f, indent=2)


    # y_pred sobre el set de validación
    y_pred = best_model.predict(X_val)

    # reporte por clase
    report_dict = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    Path("reports").mkdir(parents=True, exist_ok=True)
    report_df.to_csv("reports/classification_report.csv", index=True)
    # Guarda el mejor modelo
    joblib.dump(best_model, "models/best_model.joblib")

    print("Entrenamiento finalizado.")
    print("Métricas (validación):", json.dumps(metrics_val, indent=2))


if __name__ == "__main__":
    main()
