import os, json, joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

TRAIN = Path("data/interim/train_prepared.csv")
TEST  = Path("data/interim/test_prepared.csv")
REPORTS_DIR = Path("reports/experiments"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR  = Path("models"); MODELS_DIR.mkdir(exist_ok=True, parents=True)

def load_xy(path: Path):
    df = pd.read_csv(path)
    target = "NObeyesdad" if "NObeyesdad" in df.columns else df.columns[-1]
    X, y = df.drop(columns=[target]), df[target]
    return X, y, target

def main():
    assert TRAIN.exists() and TEST.exists(), "Faltan archivos interim (corre: dvc repro)"
    Xtr, ytr, target = load_xy(TRAIN)
    Xte, yte, _      = load_xy(TEST)

    # Modelo base + grid de hiperparámetros (sencillo y rápido)
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3]
    }
    grid = GridSearchCV(model, param_grid=param_grid, scoring="f1_macro", cv=3, n_jobs=-1, verbose=1)
    grid.fit(Xtr, ytr)

    best = grid.best_estimator_
    y_pred = best.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, y_pred, average="macro", zero_division=0)

    # Guardar resultados del experimento
    exp = {
        "model": "GradientBoostingClassifier",
        "best_params": grid.best_params_,
        "cv_best_score_f1_macro": float(grid.best_score_),
        "test_metrics": {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1)
        }
    }
    with open(REPORTS_DIR / "gb_gridsearch.json", "w") as f:
        json.dump(exp, f, indent=2)

    # Guardar matriz de confusión del experimento
    cm = confusion_matrix(yte, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "gb_confusion_matrix.png")
    plt.close()

    # Guardar el mejor estimador (no sustituye al modelo oficial del pipeline)
    joblib.dump(best, MODELS_DIR / "gb_best_grid.pkl")
    print("[OK] reports/experiments/gb_gridsearch.json y gb_confusion_matrix.png, models/gb_best_grid.pkl")

if __name__ == "__main__":
    main()