# -*- coding: utf-8 -*-
"""
Entrenamiento con selección automática de hiperparámetros (Grid/Random) y comparación de modelos.

Resumen:
- Lee datos preprocesados desde data/interim/*.csv
- Modelos: LogisticRegression, RandomForest, HistGradientBoosting, SVC (RBF)
- Búsqueda de hiperparámetros AUTO: si el grid de un modelo es grande -> RandomizedSearchCV; si es pequeño -> GridSearchCV
- Métrica principal: f1_macro (adecuada para desbalance)
- Artefactos:
    models/best_model.joblib
    reports/metrics.json
    reports/classification_report.csv
    reports/figures/confusion_matrix.png
    reports/final_model_comparison.csv
"""

import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import joblib
from joblib import dump

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# =========================
# Configuración
# =========================
CFG = {
    "paths": {
        "X_train": "data/interim/train_prepared.csv",
        "X_test": "data/interim/test_prepared.csv",
        "reports_dir": "reports",
        "figures_dir": "reports/figures",
        "models_dir": "models",
    },
    "cv": {"n_splits": 5, "random_state": 42, "shuffle": True},
    "search": {
        "mode": "auto",
        "max_combinations": 250,
        "n_iter": 40,
        "scoring": "f1_macro",
        "n_jobs": -1,
        "verbose": 1,
        "per_model": {
            "logistic_regression": "auto",
            "random_forest": "auto",
            "hist_gradient_boosting": "auto",
            "svc_rbf": "auto"
        }
    },
    "models": {
        "logistic_regression": True,
        "random_forest": True,
        "hist_gradient_boosting": True,
        "svc_rbf": True
    },
    "grids": {
        "logistic_regression": {
            "C": [0.5, 1.0, 2.0, 5.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [200, 400],
            "class_weight": [None, "balanced"],
        },
        "random_forest": {
            "n_estimators": [300, 600, 1000],
            "max_depth": [None, 12, 24, 36],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced_subsample"],
        },
        "hist_gradient_boosting": {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [None, 8, 12],
            "max_leaf_nodes": [31, 63, 127],
            "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0.0, 0.1, 0.5],
        },
        "svc_rbf": {
            "C": [0.5, 1.0, 2.0, 5.0],
            "gamma": ["scale", 0.1, 0.01],
            "probability": [True],
        },
    },
}

# =========================
# Utilidades
# =========================
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def ensure_dirs(root: Path, cfg: dict):
    for d in [cfg["paths"]["reports_dir"], cfg["paths"]["figures_dir"], cfg["paths"]["models_dir"]]:
        (root / d).mkdir(parents=True, exist_ok=True)

def read_data(root: Path, cfg: dict):
    """Lee los datos desde data/interim y separa X/y automáticamente."""
    train_path = root / cfg["paths"]["X_train"]
    test_path = root / cfg["paths"]["X_test"]

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Detectar la variable objetivo
    target_candidates = ["NObeyesdad", "target", "label", "y"]
    target_col = next((c for c in target_candidates if c in df_train.columns), df_train.columns[-1])

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col].astype(str)
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].astype(str)

    return X_train, X_test, y_train, y_test

def get_cv(cfg: dict):
    return StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"]
    )

def count_param_combinations(grid: Dict[str, list]) -> int:
    total = 1
    for values in grid.values():
        total *= len(values) if isinstance(values, (list, tuple)) else 1
    return total

def choose_mode_for_model(global_mode: str, per_model: dict, model_name: str) -> str:
    m = (per_model or {}).get(model_name)
    return m if m in {"auto", "grid", "random"} else global_mode

def make_search(estimator, grid, cfg, cv, model_name: str):
    scoring = cfg["search"]["scoring"]
    n_jobs = cfg["search"]["n_jobs"]
    verbose = cfg["search"]["verbose"]
    n_iter = cfg["search"]["n_iter"]
    max_combos = cfg["search"]["max_combinations"]
    global_mode = cfg["search"]["mode"]
    per_model = cfg["search"].get("per_model", {})

    mode = choose_mode_for_model(global_mode, per_model, model_name)
    n_combos = count_param_combinations(grid)
    chosen = "grid" if (mode == "auto" and n_combos <= max_combos) else "random"

    print(f" -> [{model_name}] grid combinations: {n_combos} | mode: {chosen}")

    if chosen == "random":
        return RandomizedSearchCV(
            estimator, grid, n_iter=min(n_iter, n_combos),
            scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose,
            refit=True, random_state=cfg["cv"]["random_state"]
        )
    return GridSearchCV(estimator, grid, scoring=scoring, cv=cv,
                        n_jobs=n_jobs, verbose=verbose, refit=True)

def plot_confusion_matrix(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix (Test)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)

# =========================
# Modelos
# =========================
def build_models_and_grids(cfg: dict):
    use, grids_cfg = cfg["models"], cfg["grids"]
    models, grids = {}, {}
    if use["logistic_regression"]:
        models["logistic_regression"] = LogisticRegression(max_iter=400)
        grids["logistic_regression"] = grids_cfg["logistic_regression"]
    if use["random_forest"]:
        models["random_forest"] = RandomForestClassifier(random_state=cfg["cv"]["random_state"])
        grids["random_forest"] = grids_cfg["random_forest"]
    if use["hist_gradient_boosting"]:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(random_state=cfg["cv"]["random_state"])
        grids["hist_gradient_boosting"] = grids_cfg["hist_gradient_boosting"]
    if use["svc_rbf"]:
        models["svc_rbf"] = SVC()
        grids["svc_rbf"] = grids_cfg["svc_rbf"]
    return models, grids

# =========================
# Entrenamiento principal
# =========================
def main():
    root = project_root()
    ensure_dirs(root, CFG)
    X_train, X_test, y_train, y_test = read_data(root, CFG)
    labels_order = sorted(y_train.unique())

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, #clases: {len(labels_order)}")

    cv = get_cv(CFG)
    models, grids = build_models_and_grids(CFG)
    results, best = [], {"score": -np.inf, "name": None, "estimator": None}

    for name, model in models.items():
        print(f"\n=== Buscando hiperparámetros para: {name} ===")
        search = make_search(model, grids[name], CFG, cv, name)
        search.fit(X_train, y_train)
        dump(search.best_estimator_, root / "models" / f"{name}_best.joblib")

        y_pred = search.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        results.append({
            "model": name,
            "cv_f1_macro": search.best_score_,
            "test_f1_macro": f1,
            "best_params": search.best_params_
        })
        if f1 > best["score"]:
            best.update({"score": f1, "name": name, "estimator": search.best_estimator_})

    best_est = best["estimator"]
    joblib.dump(best_est, root / "models" / "best_model.joblib")
    y_pred_best = best_est.predict(X_test)

    metrics = {
        "best_model": best["name"],
        "test_f1_macro": float(f1_score(y_test, y_pred_best, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, y_pred_best))
    }
    with open(root / "reports" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    pd.DataFrame(results).to_csv(root / "reports" / "final_model_comparison.csv", index=False)
    plot_confusion_matrix(y_test, y_pred_best, labels_order, root / "reports" / "figures" / "confusion_matrix.png")

    print(f"\n Entrenamiento finalizado. Mejor modelo: {best['name']} ({best['score']:.4f})")

if __name__ == "__main__":
    main()
