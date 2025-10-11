#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="CSV crudo")
    p.add_argument("--output", required=True, help="CSV limpio")
    args = p.parse_args()

    inp  = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Lee datos crudos
    df = pd.read_csv(inp)

    # --- Reglas de limpieza mínimas ---
    df = df.drop_duplicates()

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna("Missing")

    df.to_csv(outp, index=False)
    print(f"[OK] Limpieza completada → {outp}")

if __name__ == "__main__":
    main()