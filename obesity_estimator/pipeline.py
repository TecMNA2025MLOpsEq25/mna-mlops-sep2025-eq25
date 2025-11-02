"""
obesity_estimator/pipeline.py  (robusto con normalización de texto)

- Normaliza strings categóricos a minúsculas y sin espacios (strip) antes de codificar.
- OneHotEncoder usa 'sparse_output=False' en sklearn nuevos (fallback a 'sparse=False' en antiguos).
- OrdinalEncoder usa categorías provistas en params.yaml pero **normalizadas** a minúsculas/strip para evitar
  desalineaciones ('Sometimes' vs 'sometimes', etc.).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC


def _ohe_kwargs():
    # sklearn >=1.2 usa sparse_output; versiones viejas aceptan sparse
    try:
        return {"handle_unknown": "ignore", "sparse_output": False}
    except TypeError:
        return {"handle_unknown": "ignore", "sparse": False}


def _to_lower_strip(X):
    """Convierte a string, recorta espacios y pasa a minúsculas. Conserva forma (n, k)."""
    # X puede ser ndarray o DataFrame
    if isinstance(X, pd.DataFrame):
        arr = X.astype(str).to_numpy()
    else:
        arr = X.astype(str)
    arr = np.char.strip(np.char.lower(arr))
    return arr


def _normalize_categories(categories):
    """Normaliza listas de categorías a minúsculas/strip (acepta dict o lista de listas)."""
    if isinstance(categories, dict):
        return {k: [str(v).strip().lower() for v in vals] for k, vals in categories.items()}
    if isinstance(categories, list):
        return [[str(v).strip().lower() for v in vals] for vals in categories]
    return categories


def _build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    bin_cols: Optional[List[str]] = None,
    ordinal_cols: Optional[List[str]] = None,
    ordinal_categories: Optional[Dict[str, List[str]]] = None,
) -> ColumnTransformer:
    bin_cols = bin_cols or []
    ordinal_cols = ordinal_cols or []
    ordinal_categories = ordinal_categories or {}

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ohe_args = _ohe_kwargs()

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("normalize", FunctionTransformer(_to_lower_strip, feature_names_out="one-to-one")),
            ("ohe", OneHotEncoder(**ohe_args)),
        ]
    )

    bin_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("normalize", FunctionTransformer(_to_lower_strip, feature_names_out="one-to-one")),
            ("ohe", OneHotEncoder(**ohe_args)),
        ]
    )

    ord_pipe = None
    if ordinal_cols:
        norm_cats_map = _normalize_categories(ordinal_categories)
        # Construir lista de categorías en el mismo orden de ordinal_cols
        if isinstance(norm_cats_map, dict):
            categories = [norm_cats_map.get(col, None) for col in ordinal_cols]
        else:
            categories = "auto"

        ord_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("normalize", FunctionTransformer(_to_lower_strip, feature_names_out="one-to-one")),
                ("ord", OrdinalEncoder(categories=categories)),
            ]
        )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if bin_cols:
        transformers.append(("bin", bin_pipe, bin_cols))
    if ordinal_cols and ord_pipe is not None:
        transformers.append(("ord", ord_pipe, ordinal_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre


def _build_classifier(model_cfg: Dict) -> BaseEstimator:
    mtype = model_cfg.get("type", "logreg")
    mparams = model_cfg.get("params", {})

    if mtype == "logreg":
        clf = LogisticRegression(max_iter=1000, **mparams)
    elif mtype == "random_forest":
        clf = RandomForestClassifier(**mparams)
    elif mtype == "hist_gb":
        clf = HistGradientBoostingClassifier(**mparams)
    elif mtype == "svc_rbf":
        clf = SVC(kernel="rbf", probability=True, **mparams)
    else:
        raise ValueError(f"Modelo no soportado: {mtype}")
    return clf


def build_pipeline(features_cfg: Dict, model_cfg: Dict) -> Pipeline:
    pre = _build_preprocessor(
        num_cols=features_cfg.get("num_cols", []),
        cat_cols=features_cfg.get("cat_cols", []),
        bin_cols=features_cfg.get("bin_cols", []),
        ordinal_cols=features_cfg.get("ordinal_cols", []),
        ordinal_categories=features_cfg.get("ordinal_categories", {}),
    )
    clf = _build_classifier(model_cfg)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe


def get_ohe_feature_names(pre: ColumnTransformer, input_feature_names: List[str]) -> List[str]:
    out_names: List[str] = []
    for name, transformer, cols in pre.transformers_:
        if transformer is None or name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            step_names = list(transformer.named_steps.keys())
            last = transformer.named_steps[step_names[-1]]
        else:
            last = transformer
        if hasattr(last, "get_feature_names_out"):
            base = np.array(cols, dtype=object)
            feats = last.get_feature_names_out(base)
            out_names.extend(feats.tolist())
        else:
            if isinstance(cols, list):
                out_names.extend(cols)
            else:
                out_names.append(cols)
    return out_names
