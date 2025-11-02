# -*- coding: utf-8 -*-
"""
predict.py
-----------------
Evalúa los modelos entrenados sobre el conjunto de prueba y genera:
  - reports/evaluation_results.csv  (métricas por modelo)
  - reports/confusion_matrix.csv    (del mejor por f1_macro)
  - reports/confusion_matrix.png    (del mejor por f1_macro)
"""

from pathlib import Path
import os
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix
)

# Config del proyecto
from obesity_estimator.config import TEST_FILEPATH, MODELS_DIR, REPORTS_DIR
from obesity_estimator.utils import evaluate_model  # se usa si está disponible

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = Path(TEST_FILEPATH) if os.path.isabs(TEST_FILEPATH) else (REPO_ROOT / TEST_FILEPATH)
MODELS_DIR = Path(MODELS_DIR) if os.path.isabs(MODELS_DIR) else (REPO_ROOT / MODELS_DIR)
REPORTS_DIR = Path(REPORTS_DIR) if os.path.isabs(REPORTS_DIR) else (REPO_ROOT / REPORTS_DIR)

EVAL_RESULTS_CSV = REPORTS_DIR / "evaluation_results.csv"
CONF_MAT_CSV = REPORTS_DIR / "confusion_matrix.csv"
CONF_MAT_PNG = REPORTS_DIR / "confusion_matrix.png"

TARGET_CANDIDATES = ["NObeyesdad", "target", "label", "y"]


def ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def split_Xy(df: pd.DataFrame):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return df.drop(columns=[c]), df[c].astype(str)
    # fallback: última columna como target
    return df.iloc[:, :-1], df.iloc[:, -1].astype(str)


def find_models():
    """
    Busca modelos en 'models/'.
    Prioridad:
      1) best_model.joblib
      2) *_best.pkl
      3) *.joblib adicionales
      4) *.pkl adicionales
    Devuelve lista de tuplas (nombre, ruta).
    """
    candidates = []

    # 1) Mejor modelo único
    best_joblib = MODELS_DIR / "best_model.joblib"
    if best_joblib.exists():
        candidates.append(("best_model", best_joblib))

    # 2) Modelos con sufijo _best.pkl (estilo original)
    for p in sorted(MODELS_DIR.glob("*_best.pkl")):
        name = p.name.replace("_best.pkl", "")
        candidates.append((name, p))

    # 3) Otros joblib (excluyendo el best_model.joblib ya agregado)
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        if p.name != "best_model.joblib":
            candidates.append((p.stem, p))

    # 4) Otros pkl
    for p in sorted(MODELS_DIR.glob("*.pkl")):
        if not p.name.endswith("_best.pkl"):
            candidates.append((p.stem, p))

    # De-duplicar por nombre preservando prioridad
    seen = set()
    unique = []
    for name, path in candidates:
        if name not in seen:
            unique.append((name, path))
            seen.add(name)

    if not unique:
        raise FileNotFoundError(
            "No se encontraron modelos en 'models/'. "
            "Asegúrate de haber corrido la etapa 'training'."
        )
    return unique


def compute_metrics(y_true, y_pred, y_proba=None):
    """Calcula métricas básicas con defaults seguros."""
    metrics = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión - Mejor modelo (test)")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    return cm_df


def main():
    ensure_dirs()

    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el dataset de prueba: {TEST_PATH}")

    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = split_Xy(test_df)
    labels_order = sorted(y_test.unique().tolist())

    model_paths = find_models()

    results = []
    per_model_preds = {}

    logger.info("=== Evaluación de modelos ===")
    for name, path in model_paths:
        logger.info(f"Evaluando modelo: {name} ({path.name})")
        model = joblib.load(path)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Intentar usar evaluate_model si existe y devuelve las llaves esperadas
        try:
            metrics = evaluate_model(y_test, y_pred, y_proba)
            # Garantizar f1_macro por si la implementación no lo devuelve
            if "f1_macro" not in metrics:
                metrics.update(compute_metrics(y_test, y_pred, y_proba))
        except Exception:
            # Fallback siempre válido
            metrics = compute_metrics(y_test, y_pred, y_proba)

        row = {"model": name, "path": path.name}
        row.update(metrics)
        results.append(row)
        per_model_preds[name] = y_pred

    # DataFrame de resultados y orden por f1_macro (si existe)
    results_df = pd.DataFrame(results)
    if "f1_macro" in results_df.columns:
        results_df = results_df.sort_values(by="f1_macro", ascending=False)

    EVAL_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(EVAL_RESULTS_CSV, index=False)
    logger.info(f"Métricas guardadas en: {EVAL_RESULTS_CSV}")

    # Mejor modelo
    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    logger.info(f"Mejor modelo: {best_name} | f1_macro={best_row.get('f1_macro', float('nan')):.4f}")

    # Matriz de confusión del mejor
    y_pred_best = per_model_preds[best_name]
    cm_df = plot_confusion_matrix(y_test, y_pred_best, labels_order, CONF_MAT_PNG)
    cm_df.to_csv(CONF_MAT_CSV)
    logger.info(f"Matriz de confusión guardada en: {CONF_MAT_PNG}")
    logger.info("Evaluación finalizada exitosamente.")


if __name__ == "__main__":
    main()
