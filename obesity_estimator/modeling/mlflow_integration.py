# obesity_estimator/mlflow_integration.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.sklearn


def _flatten(d: Dict, parent: str = "", sep: str = ".") -> Dict:
    """Aplana diccionarios anidados para log_params."""
    flat = {}
    for k, v in d.items():
        kk = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            flat.update(_flatten(v, kk, sep))
        else:
            flat[kk] = v
    return flat


def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_training_run(
    *,
    run_name: str,
    tracking_uri: str,
    experiment_name: str,
    params: Dict,
    metrics: Dict,
    artifacts: Optional[List[Path]] = None,
    model=None,
    registered_model_name: Optional[str] = None,
) -> str:
    """
    Registra un run de MLflow:
      - set_tracking_uri + set_experiment
      - log_params (aplanados)
      - log_metrics
      - log_artifacts (archivos)
      - log_model (opcional + registry opcional)

    Devuelve el run_id.
    """
    configure_mlflow(tracking_uri, experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        # Params
        mlflow.log_params(_flatten(params))

        # Métricas
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass

        # Artefactos
        if artifacts:
            for p in artifacts:
                p = Path(p)
                if p.is_file():
                    mlflow.log_artifact(str(p))
                elif p.is_dir():
                    mlflow.log_artifacts(str(p))

        # Modelo
        if model is not None:
            if registered_model_name:
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=registered_model_name,
                )
            else:
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        return run.info.run_id
