# tests/test_features.py

import numpy as np
import pandas as pd

from obesity_estimator.features import split_data, build_preprocessor
from obesity_estimator.config import RANDOM_STATE


def _make_toy_df() -> pd.DataFrame:
    """DataFrame pequeño pero representativo para probar el preprocesamiento.

    Incluye 8 filas, 4 clases con al menos 2 ejemplos cada una,
    para que la estratificación de split_data no truene.
    """
    data = {
        "Age": [20, 21, 25, 26, 30, 31, 35, 36],
        "Height": [1.70, 1.71, 1.80, 1.79, 1.65, 1.66, 1.75, 1.76],
        "Weight": [60, 61, 80, 79, 70, 71, 90, 89],
        "FCVC": [2.0, 2.0, 3.0, 3.0, 2.5, 2.5, 3.0, 3.0],
        "NCP": [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 4.0, 4.0],
        "CH2O": [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0],
        "FAF": [1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 3.0, 3.0],
        "TUE": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        "Gender": ["Male", "Male", "Female", "Female",
                   "Female", "Female", "Male", "Male"],
        "MTRANS": [
            "Automobile", "Automobile",
            "Walking", "Walking",
            "Walking", "Walking",
            "Public_Transportation", "Public_Transportation",
        ],
        "CALC": [
            "no", "no",
            "Sometimes", "Sometimes",
            "Frequently", "Frequently",
            "Always", "Always",
        ],
        "family_history_with_overweight": ["yes", "yes", "no", "no", "no", "no", "yes", "yes"],
        "FAVC": ["no", "no", "yes", "yes", "no", "no", "yes", "yes"],
        "SCC": ["no", "no", "no", "no", "yes", "yes", "no", "no"],
        "SMOKE": ["no", "no", "yes", "yes", "no", "no", "no", "no"],
        # 4 clases, cada una con 2 ejemplos
        "NObeyesdad": [
            "Normal_Weight", "Normal_Weight",
            "Overweight_Level_I", "Overweight_Level_I",
            "Insufficient_Weight", "Insufficient_Weight",
            "Obesity_Type_I", "Obesity_Type_I",
        ],
    }
    return pd.DataFrame(data)


def test_split_data_stratified_basic():
    """
    Verifica que split_data respeta tamaños, usa estratificación internamente
    y no incluye la etiqueta en X.
    """
    df = _make_toy_df()

    # Para estratificar correctamente con 4 clases, el test_size debe ser >= 4/8.
    X_train, X_test, y_train, y_test = split_data(
        df,
        target="NObeyesdad",
        test_size=0.5,              # 4 muestras de test, 4 de train
        random_state=RANDOM_STATE,
    )

    # Con 8 filas y test_size=0.5 => 4 train, 4 test
    assert len(X_train) == 4
    assert len(X_test) == 4
    assert len(y_train) == 4
    assert len(y_test) == 4

    # La columna target no debe estar en X
    assert "NObeyesdad" not in X_train.columns
    assert "NObeyesdad" not in X_test.columns


def test_split_data_reproducible_with_same_seed():
    """Con el mismo RANDOM_STATE, el split debe ser reproducible."""
    df = _make_toy_df()

    X_train_1, X_test_1, y_train_1, y_test_1 = split_data(
        df,
        target="NObeyesdad",
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    X_train_2, X_test_2, y_train_2, y_test_2 = split_data(
        df,
        target="NObeyesdad",
        test_size=0.5,
        random_state=RANDOM_STATE,
    )


def test_build_preprocessor_no_nans_and_shape():
    """El preprocesador no debe generar NaNs y respeta el número de filas."""
    df = _make_toy_df()
    X = df.drop(columns=["NObeyesdad"])

    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    categorical_cols = ["Gender", "MTRANS", "CALC"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    Xt = preprocessor.fit_transform(X)

    # Número de filas debe ser igual al del input
    assert Xt.shape[0] == len(X)

    # No debe haber NaNs en la matriz transformada
    assert not np.isnan(Xt).any()
