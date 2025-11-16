# -*- coding: utf-8 -*-
"""
data_drift.py

Comparación entre el dataset original y el dataset drifted
para identificar data drift mediante PSI, KS-test y análisis
de cambios en distribuciones.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os


def calculate_psi(expected, actual, buckets=10):
    """Calcula Population Stability Index (PSI)."""
    def to_bins(s):
        return pd.qcut(s.rank(method='first'), q=buckets, labels=False, duplicates='drop')

    expected, actual = expected.dropna(), actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    expected_bins = to_bins(expected)
    actual_bins = to_bins(actual)

    psi_value = 0
    for b in range(buckets):
        e_perc = (expected_bins == b).mean()
        a_perc = (actual_bins == b).mean()
        if e_perc == 0: e_perc = 1e-6
        if a_perc == 0: a_perc = 1e-6
        psi_value += (e_perc - a_perc) * np.log(e_perc / a_perc)

    return psi_value


def categorical_drift(expected, actual):
    """Diferencia porcentual simple entre distribuciones categóricas."""
    exp_dist = expected.value_counts(normalize=True)
    act_dist = actual.value_counts(normalize=True)
    df = pd.concat([exp_dist, act_dist], axis=1, keys=["expected", "actual"]).fillna(0)
    df["abs_diff"] = (df["expected"] - df["actual"]).abs()
    return df["abs_diff"].sum()


def main():

    print("[INFO] Cargando datasets...")
    df_base = pd.read_csv("data/processed/obesity_estimation_clean.csv")
    df_drift = pd.read_csv("data/drifted/df_drift.csv")

    numeric_cols = df_base.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_base.select_dtypes(exclude=[np.number]).columns.tolist()

    drift_report = {"numeric": {}, "categorical": {}}

    print("[INFO] Calculando drift para numéricos...")
    for col in numeric_cols:
        psi_val = calculate_psi(df_base[col], df_drift[col])
        ks_stat, ks_p = ks_2samp(df_base[col].dropna(), df_drift[col].dropna())

        drift_report["numeric"][col] = {
            "psi": float(psi_val),
            "ks_stat": float(ks_stat),
            "ks_p_value": float(ks_p),
            "drift_detected": bool((psi_val > 0.2) or (ks_p < 0.05))
        }

    print("[INFO] Calculando drift para categóricos...")
    for col in categorical_cols:
        drift_score = categorical_drift(df_base[col], df_drift[col])
        drift_report["categorical"][col] = {
            "distribution_change": float(drift_score),
            "drift_detected": bool(drift_score > 0.25)
        }

    # ----- Guardar JSON -----
    os.makedirs("reports/drift", exist_ok=True)
    json_path = "reports/drift/drift_report.json"
    with open(json_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    print(f"[INFO] Reporte JSON guardado en: {json_path}")

    # ----- Gráfica PSI -----
    psi_values = {col: data["psi"] for col, data in drift_report["numeric"].items()}

    plt.figure(figsize=(10, 5))
    plt.bar(psi_values.keys(), psi_values.values())
    plt.xticks(rotation=90)
    plt.title("PSI por Variable (Data Drift)")
    plt.ylabel("PSI")
    plt.tight_layout()

    plot_path = "reports/drift/drift_plot.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Gráfica guardada en: {plot_path}")
    print("[INFO] Data drift stage completado.")


if __name__ == "__main__":
    main()
