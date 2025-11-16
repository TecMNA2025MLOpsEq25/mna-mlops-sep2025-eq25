# obesity_estimator/modeling/search.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, Type
import inspect

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from obesity_estimator.config import RANDOM_STATE


def _filter_params(estimator_cls: Type, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve solo los parámetros aceptados por __init__ del estimador.
    Evita fallos al pasar llaves ajenas (p.ej., learning_rate a RandomForest).
    """
    if not params:
        return {}
    valid = set(inspect.signature(estimator_cls.__init__).parameters.keys())
    return {k: v for k, v in params.items() if k in valid}


def build_estimator(model_cfg: Dict[str, Any]):
    """
    Construye el estimador base a partir de params.yaml:model,
    aplicando defaults por tipo y filtrando hiperparámetros inválidos.
    Además, fija random_state cuando el estimador lo soporta.
    """
    mtype = model_cfg["type"]
    params = dict((model_cfg.get("params") or {}))

    if mtype in ("logreg", "logistic_regression"):
        # Defaults seguros para multiclase
        params.setdefault("max_iter", 1000)
        Est = LogisticRegression

    elif mtype in ("rf", "random_forest"):
        Est = RandomForestClassifier

    elif mtype in ("svc_rbf", "svc"):
        # Forzar RBF y probabilidad para métricas ROC/PR
        params.setdefault("kernel", "rbf")
        params.setdefault("probability", True)
        Est = SVC

    elif mtype in ("hist_gb", "hist_gradient_boosting", "hgb"):
        Est = HistGradientBoostingClassifier

    else:
        raise ValueError(f"Modelo no soportado en search: '{mtype}'")

    # Si el estimador acepta random_state y no se definió en params.yaml,
    # lo fijamos al RANDOM_STATE global para reproducibilidad.
    est_signature = inspect.signature(Est.__init__).parameters
    if "random_state" in est_signature and "random_state" not in params:
        params["random_state"] = RANDOM_STATE

    clean = _filter_params(Est, params)
    return Est(**clean)


def build_search_space(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve el espacio de hiperparámetros desde params.yaml:
    model.search_space[<modelo>]
    """
    mtype = model_cfg["type"]
    spaces = model_cfg.get("search_space", {}) or {}
    space = spaces.get(mtype)
    if space is None:
        raise ValueError(
            f"No se encontró 'model.search_space.{mtype}' en params.yaml. "
            "Agrega un espacio de hiperparámetros para tu modelo."
        )
    return space


def select_searcher(kind: str):
    """Devuelve la clase de buscador a usar: 'grid' o 'random'."""
    kind = (kind or "grid").lower()
    if kind == "grid":
        return GridSearchCV
    if kind == "random":
        return RandomizedSearchCV
    raise ValueError("model.search.kind debe ser 'grid' o 'random'")
