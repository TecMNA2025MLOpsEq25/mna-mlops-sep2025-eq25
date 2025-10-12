
# ### **Tecnológico de Monterrey**
# 
# #### **Maestría en Inteligencia Artificial Aplicada**
# #### **Clase**: Operaciones de Aprendizaje Automático
# #### **Docentes**: Dr. Gerardo Rodríguez Hernández | Mtro. Ricardo Valdez Hernández | Mtro. Carlos Alberto Vences Sánchez
# 
# ##### **Actividad**: Proyecto: Avance (Fase 1) - **Notebook**: EDA
# ##### **Equipo 25**:
# | Nombre | Matrícula |
# |--------|-----------|
# | Rafael Becerra García | A01796211 |
# | Andrea Xcaret Gómez Alfaro | A01796384 |
# | David Hernández Castellanos | A01795964 |
# | Juan Pablo López Sánchez | A01313663 |
# | Osiris Xcaret Saavedra Solís | A01795992 |

# ### Objetivos:
# 
# **Analisis de Requerimientos**
# **Tarea**: Analiza la problemática a resolver siguiendo la liga con la descripción del dataset asignado.
# 
# **Manipulación y preparación de datos**
# **Tarea**: Realizar tareas de Exploratory Data Analysis (EDA)  y limpieza de datos utilizando herramientas y bibliotecas específicas (Python, Pandas, DVC, Scikitlearn, etc.)
# 
# **Exploración y preprocesamiento de datos**
# **Tarea**: Explorar y preprocesar los datos para identificar patrones, tendencias y relaciones significativas.
# 
# **Versionado de datos**
# **Tarea**: Aplicar técnicas de versionado de datos para asegurar reproducibilidad y trazabilidad.
# 
# **Construcción, ajuste y evaluación de Modelos de Machine Learning**
# **Tarea**: Construir, ajustar y evaluar modelos de Machine Learning utilizando técnicas y algoritmos apropiados al problema.

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

# --- Configuración global --- #
sns.set_theme(style="whitegrid", palette="muted", context="notebook")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

DATA_PATH = "data/processed/obesity_estimation_clean.csv"
OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=== Exploratory Data Analysis ===")

    # --- Cargar Dataset --- #
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # --- Clasificación de columnas --- #
    numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    object_cols = ["Gender", "CAEC", "CALC", "MTRANS", "NObeyesdad"]
    binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
    target_col = ["NObeyesdad"]
    object_cols_no_target = [col for col in object_cols if col not in target_col]

    print("\nColumnas detectadas:")
    print(f"Numéricas: {numeric_cols}")
    print(f"Categóricas: {object_cols}")
    print(f"Binarias: {binary_cols}")
    print(f"Objetivo: {target_col}")

    # --- Estadísticas descriptivas --- #
    desc_num = df[numeric_cols].describe()
    desc_cat = df[object_cols].describe()
    desc_bin = df[binary_cols].astype("object").describe()

    desc_num.to_csv(os.path.join(OUTPUT_DIR, "numeric_summary.csv"))
    desc_cat.to_csv(os.path.join(OUTPUT_DIR, "categorical_summary.csv"))
    desc_bin.to_csv(os.path.join(OUTPUT_DIR, "binary_summary.csv"))
    print(f"Estadísticas descriptivas guardadas en {OUTPUT_DIR}")

    # --- Histogramas y boxplots --- #
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(df[col], kde=True, bins=20, ax=axes[0])
        axes[0].set_title(f"Histograma de {col}")
        sns.boxplot(x=df[col], ax=axes[1], width=0.3)
        axes[1].set_title(f"Boxplot de {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{col}.png"))
        plt.close(fig)

    # --- Matriz de correlación --- #
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_correlation.png"))
    plt.close()

    # --- Pairplot (solo si no es demasiado pesado) --- #
    try:
        sns.pairplot(df[numeric_cols], diag_kind="kde", plot_kws={"alpha": 0.5})
        plt.suptitle("Pairplot de variables numéricas", y=1.02, fontsize=14, fontweight="bold")
        plt.savefig(os.path.join(OUTPUT_DIR, "pairplot_numeric.png"))
        plt.close()
    except Exception as e:
        print(f"No se pudo generar el pairplot: {e}")

    # --- Variables categóricas --- #
    for col in object_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(y=df[col], order=df[col].value_counts().index, palette="viridis", ax=ax, hue=df[col])
        ax.set_title(f"Conteo: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cat_count_{col}.png"))
        plt.close(fig)

    # --- Variables binarias --- #
    fig, axes = plt.subplots(1, len(binary_cols), figsize=(14, 4))
    total = len(df)
    for ax, col in zip(axes, binary_cols):
        sns.countplot(x=df[col], ax=ax, palette="Set2", order=[0, 1], hue=df[col], legend=False)
        ax.set_title(f"{col}")
        ax.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "binary_counts.png"))
    plt.close(fig)

    # --- Correlación numéricas vs binarias --- #
    for num_col in numeric_cols:
        fig, axes = plt.subplots(1, len(binary_cols), figsize=(5 * len(binary_cols), 5), sharey=True)
        for i, bin_col in enumerate(binary_cols):
            sns.boxplot(x=df[bin_col], y=df[num_col], ax=axes[i], width=0.5)
            axes[i].set_title(f"{num_col} según {bin_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"num_vs_bin_{num_col}.png"))
        plt.close(fig)

    # --- Correlación con la variable objetivo --- #
    ordered_classes = [
        "insufficient_weight",
        "normal_weight",
        "overweight_level_i",
        "overweight_level_ii",
        "obesity_type_i",
        "obesity_type_ii",
        "obesity_type_iii",
    ]
    cat_type = CategoricalDtype(categories=ordered_classes, ordered=True)
    df["NObeyesdad"] = df["NObeyesdad"].astype(cat_type)

    # Boxplots numéricos por objetivo
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x="NObeyesdad", y=col, data=df, ax=axes[i])
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_title(f"{col} por categoría de NObeyesdad")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "num_vs_target.png"))
    plt.close(fig)

    # Barras apiladas categóricas
    for col in object_cols_no_target:
        ct = pd.crosstab(df[col], df["NObeyesdad"])
        ct = ct[ordered_classes]
        ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
        ax = ct_perc.plot(kind="bar", stacked=True, figsize=(8, 4))
        plt.title(f"{col} vs NObeyesdad (%)")
        plt.ylabel("Porcentaje")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cat_vs_target_{col}.png"))
        plt.close()

    print(f"\nAnálisis completado. Figuras guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
