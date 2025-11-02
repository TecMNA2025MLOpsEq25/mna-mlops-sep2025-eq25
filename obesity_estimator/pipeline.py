# obesity_estimator/pipeline.py
# -*- coding: utf-8 -*-
"""
Pipeline robusto con normalización de texto y compatibilidad HPO.

- Normaliza strings categóricos (lower + strip) antes de codificar.
- OneHotEncoder usa 'sparse_output=False' en sklearn nuevos; fallback a 'sparse=False'.
- OrdinalEncoder usa categorías provistas en params.yaml pero normalizadas (lower/strip).
- El estimador puede inyectarse externamente (build_pipeline(..., estimator=...));
  si no se pasa, se construye a partir de model_cfg.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any

import inspect
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


# ------------------------------------------------------------
# Utilidades de compatibilidad / normalización
# ------------------------------------------------------------

def _ohe_kwargs() -> Dict[str, Any]:
    """
    Devuelve kwargs compatibles para OneHotEncoder según versión de sklearn.
    sklearn >= 1.2: 'sparse_output'
    sklearn <  1.2: 'sparse'
    """
    sig = inspect.signature(OneHotEncoder.__init__)
    if "sparse_output" in sig.parameters:
        return {"handle_unknown": "ignore", "sparse_output": False}
    else:
        return {"handle_unknown": "ignore", "sparse": False}


def _to_lower_strip(X):
    """Convierte a string, recorta espacios y pasa a minúsculas sin alterar la forma."""
    if isinstance(X, pd.DataFrame):
        arr = X.astype(str).to_numpy()
    else:
        arr = X.astype(str)
    return np.char.strip(np.char.lower(arr))


def _normalize_categories(categories):
    """
    Normaliza categorías a minúsculas/strip.
    Acepta dict (col -> lista categorias) o lista de listas en el mismo orden de columnas ordinales.
    """
    if isinstance(categories, dict):
        return {k: [str(v).strip().lower() for v in vals] for k, vals in categories.items()}
    if isinstance(categories, list):
        return [[str(v).strip().lower() for v in vals] for vals in categories]
    return categories


# ------------------------------------------------------------
# Preprocesador
# ------------------------------------------------------------

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
        # normaliza el mapa de categorías y alínealo al orden de ordinal_cols
        norm_cats_map = _normalize_categories(ordinal_categories)
        if isinstance(norm_cats_map, dict):
            categories = [norm_cats_map.get(col, None) for col in ordinal_cols]
        else:
            # ya viene como lista de listas en el orden correcto
            categories = norm_cats_map or "auto"

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


# ------------------------------------------------------------
# Clasificador (fallback si no se inyecta uno externo)
# ------------------------------------------------------------

def _build_classifier(model_cfg: Dict) -> BaseEstimator:
    mtype = (model_cfg.get("type") or "logreg").lower()
    mparams = model_cfg.get("params", {}) or {}

    # alias comunes
    if mtype in ("rf", "random_forest", "randomforest"):
        mtype = "rf"
    if mtype in ("svc", "svc_rbf"):
        mtype = "svc_rbf"

    if mtype == "logreg":
        clf = LogisticRegression(max_iter=1000, **mparams)
    elif mtype == "rf":
        clf = RandomForestClassifier(**mparams)
    elif mtype == "hist_gb":
        clf = HistGradientBoostingClassifier(**mparams)
    elif mtype == "svc_rbf":
        clf = SVC(kernel="rbf", probability=True, **mparams)
    else:
        raise ValueError(f"Modelo no soportado: {model_cfg.get('type')}")
    return clf


# ------------------------------------------------------------
# API pública
# ------------------------------------------------------------

def build_pipeline(
    features_cfg: Dict,
    model_cfg: Dict,
    estimator: Optional[BaseEstimator] = None,
) -> Pipeline:
    """
    Construye el Pipeline (pre + clf).
    - Si 'estimator' se provee (p.ej., desde train.py con HPO), se usa ese.
    - En caso contrario, se construye con _build_classifier(model_cfg).
    """
    pre = _build_preprocessor(
        num_cols=features_cfg.get("num_cols", []),
        cat_cols=features_cfg.get("cat_cols", []),
        bin_cols=features_cfg.get("bin_cols", []),
        ordinal_cols=features_cfg.get("ordinal_cols", []),
        ordinal_categories=features_cfg.get("ordinal_categories", {}),
    )
    clf = estimator if estimator is not None else _build_classifier(model_cfg)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe


def get_ohe_feature_names(pre: ColumnTransformer, input_feature_names: List[str]) -> List[str]:
    """
    Retorna nombres de features salientes del preprocesador.
    Considera transformadores con get_feature_names_out; si no, regresa las columnas originales.
    """
    out_names: List[str] = []
    for name, transformer, cols in pre.transformers_:
        if transformer is None or name == "remainder":
            continue

        # Determina el "último" paso de cada subpipeline
        last = transformer
        if hasattr(transformer, "named_steps"):
            # toma el último step definido
            step_names = list(transformer.named_steps.keys())
            last = transformer.named_steps[step_names[-1]]

        if hasattr(last, "get_feature_names_out"):
            base = np.array(cols, dtype=object)
            feats = last.get_feature_names_out(base)
            out_names.extend(feats.tolist())
        else:
            # sin método: intenta devolver las columnas originales
            if isinstance(cols, list):
                out_names.extend(cols)
            else:
                out_names.append(cols)
    return out_names
