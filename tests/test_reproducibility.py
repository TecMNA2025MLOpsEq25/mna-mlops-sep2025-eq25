# tests/test_reproducibility.py

import joblib
import pandas as pd

from obesity_estimator.config import DATA_PROCESSED, MODELS_DIR
from obesity_estimator.modeling.train_all import _eval


def _load_artifacts():
    """Carga dataset procesado y modelo entrenado."""
    df = pd.read_csv(DATA_PROCESSED / "obesity_estimation_clean.csv")
    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"].astype(str)

    model = joblib.load(MODELS_DIR / "best_model.joblib")
    return X, y, model


def test_model_evaluation_is_reproducible():
    """
    Verifica que la evaluación del modelo es reproducible:
    dos evaluaciones consecutivas deben producir exactamente
    las mismas métricas.
    """
    X, y, model = _load_artifacts()

    metrics1 = _eval(model, X, y)
    metrics2 = _eval(model, X, y)

    # Las métricas deben ser exactamente iguales
    assert metrics1 == metrics2, (
        f"Las métricas deben ser reproducibles, pero difieren:\n"
        f"{metrics1}\n{metrics2}"
    )
