# -*- coding: utf-8 -*-
"""
obesity_estimator/api/test_api.py

Pruebas del endpoint /predict del servicio FastAPI.
Incluye casos válidos para distintas clases y casos inválidos para manejo de errores.
"""

import json
import pytest
from fastapi.testclient import TestClient
from obesity_estimator.api.app import app

client = TestClient(app)

# ----------------------------------------------------------------------
# Casos de prueba válidos (uno por clase esperada)
# ----------------------------------------------------------------------

valid_cases = [
    # Insufficient_Weight
    {
        "Gender": "Female", "Age": 20.0, "Height": 1.70, "Weight": 48.0,
        "family_history_with_overweight": "no", "FAVC": "no", "FCVC": 3.0,
        "NCP": 3.0, "CAEC": "no", "SMOKE": "no", "CH2O": 3.0,
        "SCC": "no", "FAF": 3.0, "TUE": 1.0, "CALC": "no", "MTRANS": "Walking"
    },
    # Normal_Weight
    {
        "Gender": "Female", "Age": 21.0, "Height": 1.62, "Weight": 64.0,
        "family_history_with_overweight": "yes", "FAVC": "no", "FCVC": 2.0,
        "NCP": 3.0, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2.0,
        "SCC": "no", "FAF": 0.0, "TUE": 1.0, "CALC": "no", "MTRANS": "Public_Transportation"
    },
    # Overweight_Level_I
    {
        "Gender": "Male", "Age": 27.0, "Height": 1.8, "Weight": 87.0,
        "family_history_with_overweight": "no", "FAVC": "no", "FCVC": 3.0,
        "NCP": 3.0, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2.0,
        "SCC": "no", "FAF": 2.0, "TUE": 0.0, "CALC": "Frequently", "MTRANS": "Walking"
    },
    # Overweight_Level_II
    {
        "Gender": "Male", "Age": 22.0, "Height": 1.78, "Weight": 89.8,
        "family_history_with_overweight": "no", "FAVC": "no", "FCVC": 2.0,
        "NCP": 1.0, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2.0,
        "SCC": "no", "FAF": 0.0, "TUE": 0.0, "CALC": "Sometimes", "MTRANS": "Public_Transportation"
    },
    # Obesity_Type_I
    {
        "Gender": "Male", "Age": 35.0, "Height": 1.70, "Weight": 105.0,
        "family_history_with_overweight": "yes", "FAVC": "yes", "FCVC": 1.0,
        "NCP": 4.0, "CAEC": "Frequently", "SMOKE": "no", "CH2O": 1.0,
        "SCC": "no", "FAF": 0.0, "TUE": 0.0, "CALC": "Always", "MTRANS": "Automobile"
    },
    # Obesity_Type_II
    {
        "Gender": "Male", "Age": 40.0, "Height": 1.68, "Weight": 120.0,
        "family_history_with_overweight": "yes", "FAVC": "yes", "FCVC": 1.0,
        "NCP": 4.0, "CAEC": "Always", "SMOKE": "no", "CH2O": 1.0,
        "SCC": "no", "FAF": 0.0, "TUE": 0.0, "CALC": "Always", "MTRANS": "Automobile"
    },
    # Obesity_Type_III
    {
        "Gender": "Female", "Age": 20.0, "Height": 1.65, "Weight": 165.0,
        "family_history_with_overweight": "yes", "FAVC": "yes", "FCVC": 3.0,
        "NCP": 3.0, "CAEC": "Always", "SMOKE": "no", "CH2O": 1.0,
        "SCC": "no", "FAF": 1.9, "TUE": 2.0, "CALC": "Sometimes", "MTRANS": "Public_Transportation"
    }
]


@pytest.mark.parametrize("payload", valid_cases)
def test_predict_valid_cases(payload):
    """Prueba varias combinaciones válidas de entrada."""
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Falló con payload: {payload}"
    result = response.json()
    prediction = result.get("prediction", "").lower()
    print(f"→ Predicción devuelta: {prediction}")
    assert prediction, "No se devolvió ninguna predicción"
    assert "probabilities" in result, "Faltan probabilidades en la respuesta"
    assert isinstance(result["probabilities"], dict), "El campo 'probabilities' no es un dict"


# ----------------------------------------------------------------------
# Casos de prueba inválidos (errores esperados)
# ----------------------------------------------------------------------

invalid_cases = [
    # Falta campo obligatorio
    {
        "Gender": "Male", "Age": 25.0, "Height": 1.75,
        # Falta Weight
        "family_history_with_overweight": "yes", "FAVC": "no",
        "FCVC": 2.0, "NCP": 3.0, "CAEC": "Sometimes", "SMOKE": "no",
        "CH2O": 2.0, "SCC": "no", "FAF": 1.0, "TUE": 1.0,
        "CALC": "Sometimes", "MTRANS": "Walking"
    },
    # Tipo incorrecto
    {
        "Gender": "Male", "Age": "twenty", "Height": "one.seventy",
        "Weight": "seventy", "family_history_with_overweight": "yes",
        "FAVC": "no", "FCVC": 2.0, "NCP": 3.0, "CAEC": "Sometimes",
        "SMOKE": "no", "CH2O": 2.0, "SCC": "no", "FAF": 1.0,
        "TUE": 1.0, "CALC": "Sometimes", "MTRANS": "Walking"
    },
    # Categoría inválida
    {
        "Gender": "Alien", "Age": 25.0, "Height": 1.70, "Weight": 70.0,
        "family_history_with_overweight": "yes", "FAVC": "no",
        "FCVC": 2.0, "NCP": 3.0, "CAEC": "Sometimes", "SMOKE": "no",
        "CH2O": 2.0, "SCC": "no", "FAF": 1.0, "TUE": 1.0,
        "CALC": "Sometimes", "MTRANS": "Teleportation"
    },
]


@pytest.mark.parametrize("payload", invalid_cases)
def test_predict_invalid_cases(payload):
    """Prueba casos con errores esperados (datos incompletos o inválidos)."""
    response = client.post("/predict", json=payload)
    assert response.status_code in (400, 422), f"Esperaba error, obtuvo {response.status_code}"
    print(f"✓ Error esperado con payload inválido ({response.status_code})")
