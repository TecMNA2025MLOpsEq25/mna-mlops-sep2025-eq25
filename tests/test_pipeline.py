# tests/test_pipeline.py

import numpy as np
import pandas as pd

from obesity_estimator.pipeline import _build_preprocessor


def _make_df_for_pipeline() -> pd.DataFrame:
    """
    DataFrame mínimo para probar el pipeline de features.

    Incluye numéricas, categóricas y binarias con variaciones de mayúsculas/espacios
    para validar la normalización (_to_lower_strip + OneHotEncoder).
    """
    data = [
        {
            # Fila 1 (forma "limpia")
            "Age": 25,
            "Height": 1.75,
            "Weight": 70,
            "FCVC": 2.0,
            "NCP": 3.0,
            "CH2O": 2.0,
            "FAF": 1.0,
            "TUE": 1.0,
            "Gender": "Male",
            "MTRANS": "Public_Transportation",
            "CALC": "Sometimes",
            "family_history_with_overweight": "yes",
            "FAVC": "no",
            "SCC": "no",
            "SMOKE": "no",
        },
        {
            # Fila 2: misma info, pero con espacios y casing distinto
            "Age": 25,
            "Height": 1.75,
            "Weight": 70,
            "FCVC": 2.0,
            "NCP": 3.0,
            "CH2O": 2.0,
            "FAF": 1.0,
            "TUE": 1.0,
            "Gender": " male ",                      # espacios + minúsculas
            "MTRANS": " public_transportation ",     # espacios + minúsculas
            "CALC": " sometimes ",                   # espacios + minúsculas
            "family_history_with_overweight": " YES ",  # espacios + mayúsculas
            "FAVC": " No ",
            "SCC": " no ",
            "SMOKE": " NO ",
        },
    ]
    return pd.DataFrame(data)


def test_pipeline_normalizes_categorical_and_binary_features():
    """
    Verifica que el preprocesador del pipeline trate como equivalentes
    categorías con distinta capitalización/espacios.
    """
    df = _make_df_for_pipeline()

    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    categorical_cols = ["Gender", "MTRANS", "CALC"]
    binary_cols = ["family_history_with_overweight", "FAVC", "SCC", "SMOKE"]

    pre = _build_preprocessor(
        num_cols=numeric_cols,
        cat_cols=categorical_cols,
        bin_cols=binary_cols,
        ordinal_cols=None,
        ordinal_categories=None,
    )

    Xt = pre.fit_transform(df)

    # Debe haber 2 filas (una por registro original)
    assert Xt.shape[0] == 2

    # Las dos filas deberían producir exactamente el mismo vector de features,
    # ya que difieren solo en espacios/capitalización en columnas categóricas/binarias.
    first, second = Xt[0], Xt[1]
    assert np.allclose(first, second)


def test_pipeline_output_has_no_nans_and_correct_rows():
    """
    Verifica que la salida del preprocesador:
    - No contiene NaNs.
    - Conserva el número de filas original.
    """
    df = _make_df_for_pipeline()

    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    categorical_cols = ["Gender", "MTRANS", "CALC"]
    binary_cols = ["family_history_with_overweight", "FAVC", "SCC", "SMOKE"]

    pre = _build_preprocessor(
        num_cols=numeric_cols,
        cat_cols=categorical_cols,
        bin_cols=binary_cols,
        ordinal_cols=None,
        ordinal_categories=None,
    )

    Xt = pre.fit_transform(df)

    # Mismo número de filas
    assert Xt.shape[0] == len(df)

    # Sin NaNs
    assert not np.isnan(Xt).any()
