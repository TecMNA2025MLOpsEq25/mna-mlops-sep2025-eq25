# -*- coding: utf-8 -*-
"""
data_drift.py

Comparación entre el dataset original y el dataset drifted
para identificar data drift mediante:
- PSI (Population Stability Index)
- KS-test (Kolmogorov–Smirnov)
- Cambios en distribuciones categóricas

Se asume:
- Dataset base   : data/processed/obesity_estimation_clean.csv
- Dataset drifted: data/drifted/df_drift.csv

Salida:
- reports/drift/drift_report.json
- reports/drift/numeric_drift_summary.csv
- reports/drift/categorical_drift_summary.csv
- reports/drift/drift_plot.png
"""

import json
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calcula el Population Stability Index (PSI) entre dos variables numéricas.

    IMPORTANTE:
    - Los bins se definen SOLO con la distribución base (expected).
    - Esos mismos bins se aplican al conjunto drifted (actual).
    - Esto permite capturar diferencias reales de distribución.

    PSI = Σ (e_i - a_i) * ln(e_i / a_i)

    Donde:
    - e_i: proporción en el bin i del baseline
    - a_i: proporción en el bin i del conjunto drifted
    """
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Definir bins con el baseline usando quantiles
    try:
        # pd.qcut devuelve las etiquetas y los bordes; usamos solo los bordes
        _, bin_edges = pd.qcut(expected, q=buckets, retbins=True, duplicates="drop")
    except ValueError:
        # Por ejemplo, si todos los valores son iguales o no se pueden formar quantiles
        return 0.0

    # Cortar ambas series con los mismos bins
    expected_bins = pd.cut(expected, bins=bin_edges, include_lowest=True)
    actual_bins = pd.cut(actual, bins=bin_edges, include_lowest=True)

    psi_value = 0.0
    # Recorremos cada categoría (bin)
    for b in expected_bins.cat.categories:
        e_perc = (expected_bins == b).mean()
        a_perc = (actual_bins == b).mean()

        # Evitar division by zero / log(0)
        if e_perc == 0:
            e_perc = 1e-6
        if a_perc == 0:
            a_perc = 1e-6

        psi_value += (e_perc - a_perc) * np.log(e_perc / a_perc)

    return float(psi_value)


def analyze_numeric_drift(
    df_base: pd.DataFrame,
    df_drift: pd.DataFrame,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Calcula PSI y KS-test para todas las columnas numéricas compartidas
    entre df_base y df_drift.

    Devuelve un diccionario:
    {
      col: {
        "psi": ...,
        "ks_stat": ...,
        "ks_p_value": ...,
        "drift_detected": bool
      },
      ...
    }
    """
    numeric_cols = df_base.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c in df_drift.columns]

    numeric_drift: Dict[str, Any] = {}

    for col in numeric_cols:
        base_col = df_base[col]
        drift_col = df_drift[col]

        psi_val = calculate_psi(base_col, drift_col)
        ks_stat, ks_p = ks_2samp(base_col.dropna(), drift_col.dropna())

        drift_flag = bool((psi_val > psi_threshold) or (ks_p < ks_alpha))

        numeric_drift[col] = {
            "psi": float(psi_val),
            "ks_stat": float(ks_stat),
            "ks_p_value": float(ks_p),
            "drift_detected": drift_flag,
        }

        print(
            f"[NUMERIC DRIFT] {col}: "
            f"PSI={psi_val:.4f}, KS_stat={ks_stat:.4f}, KS_p={ks_p:.3e}, "
            f"drift_detected={drift_flag}"
        )

    return numeric_drift


def analyze_categorical_drift(
    df_base: pd.DataFrame,
    df_drift: pd.DataFrame,
    dist_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Analiza drift en variables categóricas midiendo el cambio absoluto total
    en las distribuciones de categorías.

    Para cada columna se calcula:
    - baseline_dist: distribución normalizada en el dataset base
    - drifted_dist : distribución normalizada en el dataset drifted
    - distribution_change: suma |p_i(base) - p_i(drift)|
    - drift_detected: bool(distribution_change > dist_threshold)
    """
    categorical_cols = df_base.select_dtypes(exclude=[np.number]).columns
    categorical_cols = [c for c in categorical_cols if c in df_drift.columns]

    categorical_drift: Dict[str, Any] = {}

    for col in categorical_cols:
        base_dist = df_base[col].value_counts(normalize=True)
        drift_dist = df_drift[col].value_counts(normalize=True)

        categories = sorted(set(base_dist.index) | set(drift_dist.index))

        total_change = 0.0
        for cat in categories:
            p_base = float(base_dist.get(cat, 0.0))
            p_drift = float(drift_dist.get(cat, 0.0))
            total_change += abs(p_base - p_drift)

        drift_flag = bool(total_change > dist_threshold)

        categorical_drift[col] = {
            "baseline_dist": {str(k): float(v) for k, v in base_dist.to_dict().items()},
            "drifted_dist": {str(k): float(v) for k, v in drift_dist.to_dict().items()},
            "distribution_change": float(total_change),
            "drift_detected": drift_flag,
        }

        print(
            f"[CATEGORICAL DRIFT] {col}: "
            f"distribution_change={total_change:.4f}, drift_detected={drift_flag}"
        )

    return categorical_drift


def save_drift_report(
    numeric_drift: Dict[str, Any],
    categorical_drift: Dict[str, Any],
    output_dir: str = "reports/drift",
) -> None:
    """
    Guarda:
    - JSON con todo el reporte.
    - CSVs de resumen para numéricas y categóricas.
    """
    os.makedirs(output_dir, exist_ok=True)

    drift_report = {"numeric": numeric_drift, "categorical": categorical_drift}

    json_path = os.path.join(output_dir, "drift_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Drift report guardado en: {json_path}")

    # Resumen numérico a CSV
    if numeric_drift:
        numeric_df = pd.DataFrame.from_dict(numeric_drift, orient="index")
        numeric_csv_path = os.path.join(output_dir, "numeric_drift_summary.csv")
        numeric_df.to_csv(numeric_csv_path, index_label="feature")
        print(f"[INFO] Resumen de drift numérico guardado en: {numeric_csv_path}")

    # Resumen categórico a CSV
    if categorical_drift:
        cat_rows = []
        for col, info in categorical_drift.items():
            cat_rows.append(
                {
                    "feature": col,
                    "distribution_change": info["distribution_change"],
                    "drift_detected": info["drift_detected"],
                }
            )
        categorical_df = pd.DataFrame(cat_rows)
        categorical_csv_path = os.path.join(
            output_dir, "categorical_drift_summary.csv"
        )
        categorical_df.to_csv(categorical_csv_path, index=False)
        print(
            f"[INFO] Resumen de drift categórico guardado en: "
            f"{categorical_csv_path}"
        )


def plot_psi_bar(
    numeric_drift: Dict[str, Any],
    output_dir: str = "reports/drift",
    title: str = "PSI por Variable (Data Drift)",
) -> None:
    """
    Genera una gráfica de barras con el PSI por variable numérica.
    """
    if not numeric_drift:
        print("[WARN] No hay información numérica de drift para graficar PSI.")
        return

    os.makedirs(output_dir, exist_ok=True)

    psi_values = {feat: info["psi"] for feat, info in numeric_drift.items()}
    psi_series = pd.Series(psi_values).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    psi_series.plot(kind="bar")
    plt.title(title)
    plt.ylabel("PSI")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "drift_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Gráfica de PSI guardada en: {plot_path}")


def main() -> None:
    # Rutas de entrada
    base_path = "data/processed/obesity_estimation_clean.csv"
    drift_path = "data/drifted/df_drift.csv"

    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"No se encontró el dataset base en: {base_path}. "
            "Asegúrate de haber corrido el stage de preparación de datos."
        )

    if not os.path.exists(drift_path):
        raise FileNotFoundError(
            f"No se encontró el dataset drifted en: {drift_path}. "
            "Asegúrate de haber generado df_drift.csv."
        )

    print(f"[INFO] Cargando dataset base desde:   {base_path}")
    print(f"[INFO] Cargando dataset drifted desde: {drift_path}")

    df_base = pd.read_csv(base_path)
    df_drift = pd.read_csv(drift_path)

    print("[INFO] Analizando drift numérico...")
    numeric_drift = analyze_numeric_drift(df_base, df_drift)

    print("[INFO] Analizando drift categórico...")
    categorical_drift = analyze_categorical_drift(df_base, df_drift)

    print("[INFO] Guardando reporte de drift...")
    save_drift_report(numeric_drift, categorical_drift)

    print("[INFO] Generando gráfica de PSI...")
    plot_psi_bar(numeric_drift)

    print("[INFO] Data drift stage completado.")


if __name__ == "__main__":
    main()
