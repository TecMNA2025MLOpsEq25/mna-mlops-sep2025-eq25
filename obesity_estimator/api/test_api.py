# -*- coding: utf-8 -*-
"""
obesity_estimator/api/test_api.py

Prueba local del endpoint /predict.
Lanza la app en modo test y envía una petición POST.
"""

import json
from fastapi.testclient import TestClient
from obesity_estimator.api.app import app

client = TestClient(app)


def test_predict():
    """Envía un ejemplo de datos válidos al endpoint /predict."""
    payload = {
        "Gender": "Female",
        "Age": 21.0,
        "Height": 1.62,
        "Weight": 64.0,
        "family_history_with_overweight": "yes",
        "FAVC": "no",
        "FCVC": 2.0,
        "NCP": 3.0,
        "CAEC": "Sometimes",
        "SMOKE": "no",
        "CH2O": 2.0,
        "SCC": "no",
        "FAF": 0.0,
        "TUE": 1.0,
        "CALC": "no",
        "MTRANS": "Public_Transportation"
    }

    response = client.post("/predict", json=payload)
    print("Status code:", response.status_code)
    print("Response JSON:", json.dumps(response.json(), indent=2, ensure_ascii=False))

    assert response.status_code == 200, "La respuesta no fue exitosa"
    assert "prediction" in response.json(), "No se devolvió el campo 'prediction'"


if __name__ == "__main__":
    # Permite ejecutar el test directamente sin pytest
    test_predict()
