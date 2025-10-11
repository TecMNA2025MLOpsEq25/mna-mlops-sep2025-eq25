#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def save_cm(y_true, y_pred, out_png: Path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x_train", required=True)
    p.add_argument("--x_test", required=True)
    p.add_argument("--y_train", required=True)
    p.add_argument("--y_test", required=True)
    p.add_argument("--out_model", required=True)
    p.add_argument("--out_metrics", required=True)
    p.add_argument("--out_figdir", required=True)
    args = p.parse_args()

    # Leer features y labels
    X_train = pd.read_csv(args.x_train)
    X_test  = pd.read_csv(args.x_test)
    y_train = pd.read_csv(args.y_train).iloc[:, 0]
    y_test  = pd.read_csv(args.y_test).iloc[:, 0]

    # --- Limpiar NaN en y y alinear X/Y ---
    # (si existieran NaN residuales en y, los quitamos y mantenemos la misma fila en X)
    if y_train.isna().any():
        mask = ~y_train.isna()
        X_train = X_train.loc[mask].reset_index(drop=True)
        y_train = y_train.loc[mask].reset_index(drop=True)
    if y_test.isna().any():
        mask = ~y_test.isna()
        X_test = X_test.loc[mask].reset_index(drop=True)
        y_test = y_test.loc[mask].reset_index(drop=True)

    # Forzar numérico por seguridad en X
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test  = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Modelo (baseline)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Métricas
    metrics = {
        "task": "classification",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    # Figuras
    figdir = Path(args.out_figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    cm_png = figdir / "confusion_matrix.png"
    save_cm(y_test, y_pred, cm_png)
    metrics["confusion_matrix_png"] = str(cm_png)

    # Guardar modelo
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out_model)

    # Guardar métricas
    outm = Path(args.out_metrics)
    outm.parent.mkdir(parents=True, exist_ok=True)
    with open(outm, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[OK] Training completado")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()