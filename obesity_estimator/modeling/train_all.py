# obesity_estimator/modeling/train_all.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import joblib
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report
)

from obesity_estimator.pipeline import build_pipeline
from obesity_estimator.modeling.search import build_estimator, select_searcher


def _ensure_dirs():
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/hpo").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)


def _load_dataset(processed_path: str, target: str):
    df = pd.read_csv(processed_path)
    df = df[~df[target].isna()].reset_index(drop=True)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def _safe_stratify(y: pd.Series, enable: bool):
    if not enable:
        return None
    vc = y.value_counts(dropna=False)
    if len(vc) < 2 or (vc < 2).any():
        return None
    return y


def _eval(model, X, y) -> Dict[str, float]:
    y_pred = model.predict(X)
    res = {
        "f1_macro": float(f1_score(y, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
    }
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                res["roc_auc_ovr"] = float(roc_auc_score(y, y_proba, multi_class="ovr"))
    except Exception:
        pass
    return res


def _hpo_and_fit(pipe, X_tr, y_tr, model_type: str, model_cfg: Dict[str, Any], target_cfg: Dict[str, Any]):
    """Ejecuta HPO si está habilitado; devuelve best_estimator y best_params (limpios)."""
    search_cfg = (model_cfg or {}).get("search", {})
    if not search_cfg.get("enabled", False):
        best = pipe.fit(X_tr, y_tr)
        return best, {}

    # Espacio por modelo desde model.search_space
    spaces_all = (model_cfg or {}).get("search_space", {})
    space = spaces_all.get(model_type)
    if not space:
        # Sin espacio: fit directo
        best = pipe.fit(X_tr, y_tr)
        return best, {}

    Searcher = select_searcher(search_cfg.get("kind", "grid"))
    # Prefija con 'clf__' para referir al clasificador dentro del Pipeline
    param_grid = {f"clf__{k}": v for k, v in space.items()}

    common = dict(
        scoring=target_cfg.get("metric_primary", "f1_macro"),
        n_jobs=search_cfg.get("n_jobs", -1),
        cv=search_cfg.get("cv", 5),
        verbose=search_cfg.get("verbose", 1),
        refit=True,
    )

    if search_cfg.get("kind", "grid") == "random":
        search = Searcher(pipe, param_distributions=param_grid,
                          n_iter=search_cfg.get("n_iter", 30), **common)
    else:
        search = Searcher(pipe, param_grid=param_grid, **common)

    search.fit(X_tr, y_tr)

    # Guarda resultados CV por modelo
    pd.DataFrame(search.cv_results_).to_csv(f"reports/hpo/cv_results_{model_type}.csv", index=False)

    best = search.best_estimator_
    best_params_clean = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return best, best_params_clean


def _build_pipe_with_estimator(features_cfg: Dict[str, Any], model_cfg: Dict[str, Any], base_estimator):
    """
    Intenta construir el pipeline pasando 'estimator=...' si tu build_pipeline lo soporta.
    Si no, hace fallback: construye y luego sustituye 'clf'.
    """
    try:
        pipe = build_pipeline(features_cfg, model_cfg, estimator=base_estimator)  # firma nueva
        return pipe
    except TypeError:
        # Firma antigua: build_pipeline(features_cfg, model_cfg) y luego sustituimos el clf
        pipe = build_pipeline(features_cfg, model_cfg)
        try:
            pipe.set_params(clf=base_estimator)
        except ValueError:
            # Si el paso no se llama 'clf' por alguna refactorización, fuerza el reemplazo
            from sklearn.pipeline import Pipeline as SkPipeline
            steps = list(pipe.steps)
            # reemplaza el último paso por el estimador
            if steps and steps[-1][0] != "clf":
                steps[-1] = ("clf", base_estimator)
            else:
                steps[-1] = ("clf", base_estimator)
            pipe = SkPipeline(steps=steps)
        return pipe


def main():
    _ensure_dirs()

    cfg = yaml.safe_load(open("params.yaml")) or {}
    for k in ["data", "features", "split", "target", "model"]:
        if k not in cfg:
            raise SystemExit(f"params.yaml inválido: falta la llave '{k}'")

    data_cfg = cfg["data"]
    features_cfg = cfg["features"]
    split_cfg = cfg["split"]
    target_cfg = cfg["target"]
    model_cfg = cfg["model"]

    # Modelos a evaluar en una sola corrida
    candidates: List[str] = model_cfg.get("candidates", [model_cfg.get("type", "hist_gb")])

    X, y = _load_dataset(data_cfg["processed_path"], target_cfg["name"])
    strat = _safe_stratify(y, split_cfg.get("stratify", True))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=split_cfg.get("test_size", 0.3),
        random_state=split_cfg.get("random_state", 42),
        stratify=strat
    )

    # Métrica principal para ranking de modelos
    metric_primary = target_cfg.get("metric_primary", "f1_macro")

    rows = []
    best_overall = None
    best_overall_score = float("-inf")
    best_overall_type = None

    # Barrido por modelo
    for mtype in candidates:
        # Modelo de trabajo: copia de model_cfg con tipo forzado
        mc = dict(model_cfg)
        mc["type"] = mtype

        base_est = build_estimator(mc)
        pipe = _build_pipe_with_estimator(features_cfg, mc, base_est)

        # HPO (si aplica) + fit
        best_model, best_params = _hpo_and_fit(pipe, X_tr, y_tr, mtype, mc, target_cfg)

        # Eval (validación)
        metrics_val = _eval(best_model, X_val, y_val)

        # Artefacto por tipo
        out_path = f"models/{mtype}_best.joblib"
        joblib.dump(best_model, out_path)

        rows.append({
            "model": mtype,
            **metrics_val,
            "best_params": json.dumps(best_params, ensure_ascii=False),
            "artifact": out_path
        })

        # Selección del mejor según la métrica primaria
        score = metrics_val.get(metric_primary, -1.0)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = best_model
            best_overall_type = mtype

    # Comparativa ordenada
    comp = pd.DataFrame(rows).sort_values(by=metric_primary, ascending=False)
    comp.to_csv("reports/final_model_comparison.csv", index=False)

    # Publica el mejor como artefacto canónico
    joblib.dump(best_overall, "models/best_model.joblib")

    # Métricas del mejor (validación) como metrics.json (lo espera DVC)
    best_row = comp.iloc[0].to_dict()
    metrics_json = {
        "selected_model": best_overall_type,
        "validation_metrics": {
            "f1_macro": best_row.get("f1_macro"),
            "accuracy": best_row.get("accuracy"),
            "precision_macro": best_row.get("precision_macro"),
            "recall_macro": best_row.get("recall_macro"),
            "roc_auc_ovr": best_row.get("roc_auc_ovr"),
        },
        "best_params": json.loads(best_row.get("best_params", "{}")),
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    # classification_report (validación) para satisfacer 'training' en dvc.yaml
    y_pred_val = best_overall.predict(X_val)
    report_df = pd.DataFrame(classification_report(
        y_val, y_pred_val, output_dict=True, zero_division=0
    )).transpose()
    report_df.to_csv("reports/classification_report.csv", index=True)

    print("[OK] Comparativa guardada en reports/final_model_comparison.csv")
    print(f"[OK] Mejor modelo: {best_overall_type} ({metric_primary}={best_overall_score:.4f})")
    print("[OK] Artefacto canónico: models/best_model.joblib")


if __name__ == "__main__":
    main()
