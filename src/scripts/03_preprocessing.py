#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
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

    # --- Limpiar target antes de separar (evitar NaN en y) ---
    target = args.target
    # Trata como nulos valores "Missing", vacíos o literales de NaN
    df[target] = df[target].replace(["Missing", "", " ", "NaN", "nan", None], pd.NA)
    # Elimina filas con target nulo
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # --- Separar X y y ---
    X = df.drop(columns=[target])
    y = df[target]

    # --- Identificar columnas numéricas y categóricas ---
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # --- Pipeline de preprocesamiento ---
    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    # --- Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) < 10 else None
    )

    # --- Ajustar y transformar ---
    X_train_prep = preproc.fit_transform(X_train)
    X_test_prep = preproc.transform(X_test)

    # Asegurar salida densa por si algún transformador devuelve matriz dispersa
    if hasattr(X_train_prep, "toarray"):
        X_train_prep = X_train_prep.toarray()
    if hasattr(X_test_prep, "toarray"):
        X_test_prep = X_test_prep.toarray()

    # --- Guardar preprocesador y conjuntos ---
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)
    # Asegurar carpeta para el preprocesador (p.ej., 'models/')
    Path(args.model_preproc).parent.mkdir(parents=True, exist_ok=True)

    # Guardar matrices transformadas como CSV (simplificado)
    pd.DataFrame(X_train_prep).astype(float).to_csv(args.output_train, index=False)
    pd.DataFrame(X_test_prep).astype(float).to_csv(args.output_test, index=False)

    # Guardar etiquetas (y) alineadas con los splits usando el nombre real del target
    y_train_path = Path(args.output_train).with_name("y_train.csv")
    y_test_path = Path(args.output_test).with_name("y_test.csv")
    pd.DataFrame({target: y_train}).to_csv(y_train_path, index=False)
    pd.DataFrame({target: y_test}).to_csv(y_test_path, index=False)
    print(f"[OK] Etiquetas guardadas → {y_train_path.name}, {y_test_path.name}")
    
    # Guardar preprocesador
    joblib.dump(preproc, args.model_preproc)
    print(f"[OK] Preprocesador guardado → {args.model_preproc}")

    print(f"[OK] Preprocesamiento completado → {args.output_train}, {args.output_test}")

if __name__ == "__main__":
    main()