#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV limpio de entrada")
    parser.add_argument("--output_train", required=True, help="Archivo de salida de train")
    parser.add_argument("--output_test", required=True, help="Archivo de salida de test")
    parser.add_argument("--target", required=True, help="Nombre de la variable objetivo")
    parser.add_argument("--model_preproc", required=True, help="Ruta para guardar el preprocesador")
    args = parser.parse_args()

    inp = Path(args.input)
    df = pd.read_csv(inp)

    # --- Separar X y y ---
    target = args.target
    X = df.drop(columns=[target])
    y = df[target]

    # --- Identificar columnas numéricas y categóricas ---
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # --- Pipeline de preprocesamiento ---
    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # --- Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) < 10 else None
    )

    # --- Ajustar y transformar ---
    X_train_prep = preproc.fit_transform(X_train)
    X_test_prep = preproc.transform(X_test)

    # --- Guardar preprocesador y conjuntos ---
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)
    # Asegurar carpeta para el preprocesador (p.ej., 'models/')
    Path(args.model_preproc).parent.mkdir(parents=True, exist_ok=True)

    # Guardar matrices transformadas como CSV (simplificado)
    pd.DataFrame(X_train_prep).to_csv(args.output_train, index=False)
    pd.DataFrame(X_test_prep).to_csv(args.output_test, index=False)

    # Guardar etiquetas (y) alineadas con los splits
    pd.DataFrame(y_train).to_csv(
    Path(args.output_train).with_name("y_train.csv"), index=False, header=["target"]
    )
    pd.DataFrame(y_test).to_csv(
    Path(args.output_test).with_name("y_test.csv"), index=False, header=["target"]
    )
    print("[OK] Etiquetas guardadas → y_train.csv, y_test.csv")
    
    # Guardar preprocesador
    joblib.dump(preproc, args.model_preproc)
    print(f"[OK] Preprocesador guardado → {args.model_preproc}")

    print(f"[OK] Preprocesamiento completado → {args.output_train}, {args.output_test}")

if __name__ == "__main__":
    main()