# tests/test_training_integration.py

import os
import joblib
import pandas as pd

from obesity_estimator.modeling.train_all import _eval
from obesity_estimator.config import DATA_PROCESSED, MODELS_DIR


def test_end_to_end_model_evaluation_matches_reference():
    """
    Prueba de integración OFFLINE:
    - Carga el dataset procesado.
    - Carga el modelo entrenado.
    - Ejecuta evaluación _eval.
    - Verifica que las métricas son válidas y, si existe baseline,
      que están razonablemente cercanas.
    """

    # 1) Cargar dataset procesado
    processed_path = DATA_PROCESSED / "obesity_estimation_clean.csv"
    assert processed_path.exists(), f"Dataset procesado no encontrado en {processed_path}"

    df = pd.read_csv(processed_path)

    assert "NObeyesdad" in df.columns, "El dataset procesado debe contener la etiqueta"

    X = df.drop(columns=["NObeyesdad"])

    # Forzamos la etiqueta a tipo str para evitar mezclas float+str
    y = df["NObeyesdad"].astype(str)

    # 2) Cargar modelo entrenado
    model_path = MODELS_DIR / "best_model.joblib"
    assert model_path.exists(), f"Modelo entrenado no encontrado en {model_path}"

    model = joblib.load(model_path)

    # 3) Calcular métricas con _eval
    metrics = _eval(model, X, y)

    # Métricas básicas deben existir y estar en rango [0,1]
    for m in ["accuracy", "f1_macro"]:
        assert m in metrics, f"La métrica {m} no fue calculada"
        assert 0.0 <= metrics[m] <= 1.0, f"La métrica {m} está fuera de rango: {metrics[m]}"

    # 4) Si existe baseline de métricas, comparamos que estén razonablemente cerca
    ref_path = "reports/evaluation_results.csv"
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path).iloc[0].to_dict()

        for m in ["accuracy", "f1_macro"]:
            if m in ref:
                # Permitimos una pequeña diferencia, por temas de floating point o
                # cambios mínimos en el pipeline/datos
                diff = abs(metrics[m] - ref[m])
                assert diff < 0.2, (
                    f"Métrica {m} difiere demasiado del baseline: "
                    f"{metrics[m]:.4f} vs {ref[m]:.4f} (diff={diff:.4f})"
                )
