# tools/export_mlflow_runs.py
# -*- coding: utf-8 -*-
"""
Exporta todos los experimentos de MLflow a un archivo CSV para documentación y auditoría.

Uso:
  python tools/export_mlflow_runs.py --uri mlruns --experiment Obesity_Classification
"""

import argparse
from pathlib import Path
import mlflow
import pandas as pd


def main(uri: str, experiment: str, output: str):
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"Experimento no encontrado: {experiment}")

    print(f"📊 Leyendo runs del experimento '{experiment}' en {uri}...")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out, index=False)

    print(f"Exportado {len(runs)} runs a: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True, help="Tracking URI, e.g., mlruns")
    ap.add_argument("--experiment", required=True, help="Experiment name, e.g., Obesity_Classification")
    ap.add_argument("--output", default="reports/mlflow_runs.csv", help="Ruta de salida del CSV")
    args = ap.parse_args()

    main(args.uri, args.experiment, args.output)
