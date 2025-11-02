"""
plots.py
-----------------
Realiza el Análisis Exploratorio de Datos (EDA) sobre el dataset limpio
y guarda todos los resultados en reports/figures/eda.
"""

# ----------------- #
# --- LIBRERÍAS --- #
# ----------------- #
import os
import warnings
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

from obesity_estimator.config import (
    PROCESSED_FILEPATH, FIGURES_DIR,
    NUMERIC_COLS, CATEGORICAL_COLS, BINARY_COLS, TARGET_COL
)

# ------------------------------- #
# --- CONFIGURACIÓN GLOBAL --- #
# ------------------------------- #
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", context="notebook")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


# ---------------------------- #
# --- FUNCIONES AUXILIARES --- #
# ---------------------------- #
def ensure_figures_dir():
    """Crea la carpeta reports/figures/eda."""
    out_dir = Path(FIGURES_DIR) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def existing_cols(df: pd.DataFrame, cols: list) -> list:
    """Devuelve solo las columnas existentes."""
    return [c for c in cols if c in df.columns]


def safe_savefig(path: Path):
    """Guarda figuras de forma segura."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


# ----------------------------------------- #
# --- FUNCIONES DE ANÁLISIS DESCRIPTIVO --- #
# ----------------------------------------- #
def save_descriptive_stats(df, out_dir: Path):
    """Guarda resúmenes estadísticos básicos."""
    num = existing_cols(df, NUMERIC_COLS)
    cat = existing_cols(df, CATEGORICAL_COLS)
    bin_cols = existing_cols(df, BINARY_COLS)

    if num:
        df[num].describe().to_csv(out_dir / "numeric_summary.csv")
    if cat:
        df[cat].describe().to_csv(out_dir / "categorical_summary.csv")
    if bin_cols:
        df[bin_cols].astype("object").describe().to_csv(out_dir / "binary_summary.csv")

    print(f"Estadísticas descriptivas guardadas en {out_dir}")


# ---------------------------------- #
# --- FUNCIONES DE VISUALIZACIÓN --- #
# ---------------------------------- #
def plot_numeric_distributions(df, out_dir: Path):
    num = existing_cols(df, NUMERIC_COLS)
    for col in num:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(df[col], kde=True, bins=20, ax=axes[0])
        axes[0].set_title(f"Histograma de {col}")
        sns.boxplot(x=df[col], ax=axes[1], width=0.3)
        axes[1].set_title(f"Boxplot de {col}")
        safe_savefig(out_dir / f"dist_{col}.png")


def plot_categorical_counts(df, out_dir: Path):
    cats = existing_cols(df, CATEGORICAL_COLS)
    for col in cats:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax, palette="viridis")
        ax.set_title(f"Conteo: {col}")
        safe_savefig(out_dir / f"cat_count_{col}.png")


def plot_binary_correlations(df, out_dir: Path):
    bin_cols = existing_cols(df, BINARY_COLS)
    if not bin_cols:
        return

    # Conteo binario
    fig, axes = plt.subplots(1, len(bin_cols), figsize=(5 * len(bin_cols), 5))
    if len(bin_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, bin_cols):
        sns.countplot(x=df[col], ax=ax, palette="Set2", order=[0, 1])
        ax.set_title(f"{col}")
    safe_savefig(out_dir / "binary_counts.png")

    # Boxplots numéricos vs binarias
    for num_col in existing_cols(df, NUMERIC_COLS):
        fig, axes = plt.subplots(1, len(bin_cols), figsize=(5 * len(bin_cols), 5), sharey=True)
        if len(bin_cols) == 1:
            axes = [axes]
        for i, bin_col in enumerate(bin_cols):
            sns.boxplot(x=df[bin_col], y=df[num_col], ax=axes[i], width=0.5)
            axes[i].set_title(f"{num_col} según {bin_col}")
        safe_savefig(out_dir / f"num_vs_bin_{num_col}.png")


def plot_target_relationships(df, out_dir: Path):
    if TARGET_COL not in df.columns:
        return

    ordered_classes = [
        "insufficient_weight", "normal_weight", "overweight_level_i",
        "overweight_level_ii", "obesity_type_i", "obesity_type_ii", "obesity_type_iii"
    ]
    df[TARGET_COL] = df[TARGET_COL].astype(
        CategoricalDtype(categories=ordered_classes, ordered=True)
    )

    # Boxplots numéricos vs target
    num_cols = existing_cols(df, NUMERIC_COLS)
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x=TARGET_COL, y=col, data=df, ax=axes[i])
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_title(f"{col} por categoría de {TARGET_COL}")
    for j in range(len(num_cols), len(axes)):
        axes[j].axis("off")
    safe_savefig(out_dir / "num_vs_target.png")

    # Barras apiladas categóricas vs target
    for col in [c for c in CATEGORICAL_COLS if c != TARGET_COL]:
        ct = pd.crosstab(df[col], df[TARGET_COL])
        ct = ct[ordered_classes]
        ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
        ax = ct_perc.plot(kind="bar", stacked=True, figsize=(8, 4))
        plt.title(f"{col} vs {TARGET_COL} (%)")
        plt.ylabel("Porcentaje")
        safe_savefig(out_dir / f"cat_vs_target_{col}.png")


# ------------------------- #
# --- PROCESO PRINCIPAL --- #
# ------------------------- #
def main():
    print("=== Exploratory Data Analysis ===")
    out_dir = ensure_figures_dir()
    df = pd.read_csv(PROCESSED_FILEPATH)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    save_descriptive_stats(df, out_dir)
    plot_numeric_distributions(df, out_dir)
    plot_categorical_counts(df, out_dir)
    plot_binary_correlations(df, out_dir)
    plot_target_relationships(df, out_dir)

    print(f"\nAnálisis completado. Figuras guardadas en: {out_dir.resolve()}")


# ------------------------- #
# --- EJECUCIÓN DE MAIN --- #
# ------------------------- #
if __name__ == "__main__":
    main()
