import pandas as pd
import os

def _pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def test_processed_dataset_has_target_and_rows():
    # Intentamos con rutas comunes del repo
    processed = _pick_existing(
        "data/interim/train_prepared.csv",
        "data/processed/obesity_estimation_clean.csv"
    )
    assert processed is not None, "No se encontró dataset procesado para la prueba."

    df = pd.read_csv(processed)
    assert len(df) > 0, "El dataset procesado está vacío."
    # Target típico del proyecto
    assert "NObeyesdad" in df.columns, "Falta la columna target 'NObeyesdad'."
