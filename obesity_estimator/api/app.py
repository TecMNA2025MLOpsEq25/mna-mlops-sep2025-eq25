# -*- coding: utf-8 -*-
"""
obesity_estimator/api/app.py

Servicio FastAPI para predicción del nivel de obesidad.
- Carga el modelo entrenado (best_model.joblib)
- Expone un endpoint POST /predict
- Valida entrada con Pydantic (v2)
- Devuelve la predicción y probabilidades (si aplica)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from obesity_estimator.api.schemas import ObesityInput
import joblib
import pandas as pd
from pathlib import Path
import traceback

# --- Configuración de modelo y preprocesador ---
MODEL_PATH = Path("models/best_model.joblib")
PREPROCESSOR_PATH = Path("data/interim/preprocessor.pkl")

app = FastAPI(
    title="Obesity Estimator API",
    version="1.0.0",
    description="API para predecir el nivel de obesidad a partir de datos personales y de hábitos.",
)


# --- Cargar modelo una sola vez ---
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"No se encontró el preprocesador en {PREPROCESSOR_PATH}")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None


@app.post("/predict", summary="Predice el nivel de obesidad", response_description="Predicción del modelo")
def predict(input_data: ObesityInput):
    """
    Recibe los datos del individuo y devuelve la predicción del nivel de obesidad.
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado correctamente en el servidor.")

    try:
        # Convertir input a DataFrame
        input_df = pd.DataFrame([input_data.model_dump()])

        # Convertir variables binarias a 0/1 porque el modelo las trató como numéricas
        binary_map = {"yes": 1, "no": 0, "Yes": 1, "No": 0, True: 1, False: 0}
        binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
        input_df[binary_cols] = input_df[binary_cols].replace(binary_map)

        # Realizar predicción
        y_pred = model.predict(input_df)[0]

        # Probabilidades (si el modelo lo soporta)
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)[0]
                probabilities = {cls: float(p) for cls, p in zip(model.classes_, proba)}
            except Exception:
                pass

        return JSONResponse(
            content={
                "prediction": str(y_pred),
                "probabilities": probabilities,
                "model_path": str(MODEL_PATH),
                "model_version": "1.0.0",
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")


@app.get("/", summary="Verifica estado del servicio")
def root():
    """Endpoint raíz para verificar que la API esté activa."""
    return {"message": "Obesity Estimator API activa", "model_loaded": model is not None}

