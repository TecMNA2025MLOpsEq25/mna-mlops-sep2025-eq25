
# ### **Tecnológico de Monterrey**
# 
# #### **Maestría en Inteligencia Artificial Aplicada**
# #### **Clase**: Operaciones de Aprendizaje Automático
# #### **Docentes**: Dr. Gerardo Rodríguez Hernández | Mtro. Ricardo Valdez Hernández | Mtro. Carlos Alberto Vences Sánchez
# 
# ##### **Actividad**: Proyecto: Avance (Fase 1) **Notebook**: EDA
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



# --- Importaciones e inicializaciones --- #

from pandas.api.types import CategoricalDtype

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Opciones de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

# Configuración visual
sns.set_theme(style='whitegrid', palette='muted', context='notebook')
get_ipython().run_line_magic('matplotlib', 'inline')




# --- Cargar Dataset --- #

df = pd.read_csv('../../data/processed/obesity_estimation_clean.csv')
print('Dataset de trabajo (df)', df.shape)


# ### Exploración inicial
# Revisar información general, tipos de datos, primeros registros y estadísticas descriptivas.



# --- Revisión inicial --- #

print(df.dtypes)
print(df.head())


# ### Clasificación de Columnas



# --- Separar columnas por tipo --- #

# Columnas numéricas
numeric_cols = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']

# Columnas de tipo texto
object_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad']
target_col = ['NObeyesdad']
object_cols_no_target = [col for col in object_cols if col not in target_col]

# Columnas binarias (0/1)
binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

print(f'Variables numéricas: {numeric_cols}')
print(f'Variables categóricas: {object_cols}')
print(f'Variables binarias: {binary_cols}')
print(f'Variables Objetivo: {target_col}')


# ### Estadísticas Descriptivas



# --- Estadísticas descriptivas de las columnas --- #

print("\n--- Estadísticas numéricas ---")
print(df[numeric_cols].describe())

print("\n--- Estadísticas categóricas ---")
print(df[object_cols].describe())

print("\n--- Estadísticas binarias ---")
print(df[binary_cols].astype('object').describe())


# ### Inspección Visual
# Visualización de los datos por tipo de columna



# --- Visualización de las columnas numéricas --- #

for col in numeric_cols:
    plt.figure(figsize=(14,4))

    # Histograma
    plt.subplot(1,2,1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Histograma de {col}')

    # Boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=df[col], width=0.3)
    plt.title(f'Boxplot de {col}')

    plt.tight_layout()
    plt.show()




# --- Heatmap de columnas numéricas --- #

plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()




# --- Pairplot de columnas numéricas --- #

sns.pairplot(df[numeric_cols], diag_kind="kde", plot_kws={"alpha":0.5})
plt.suptitle("Pairplot de variables numéricas", y=1.02, fontsize=14, fontweight="bold")
plt.show()




# --- Visualización de las columnas de texto --- #

for col in object_cols:
    categorias = df[col].value_counts()
    n_cats = len(categorias)

    # Ajustar altura dinámicamente
    plt.figure(figsize=(8, 2 + n_cats*0.5))

    ax = sns.countplot(
        y=df[col], 
        order=categorias.index,
        palette="viridis",
        hue=df[col]
    )

    # Título y ejes
    plt.title(f'Conteo: {col}')
    plt.xlabel('Cantidad')
    plt.ylabel('')

    # Hacer líneas verticales punteadas
    ax.grid(axis='x', linestyle=':', alpha=0.7)

    # Calcular etiquetas: conteo + porcentaje
    total = len(df)
    for p in ax.patches:
        count = int(p.get_width())
        percent = 100 * count / total
        ax.text(
            p.get_width() + 10,
            p.get_y() + p.get_height()/2, 
            f'{percent:.1f}% ({count})',
            va='center',
            fontweight='bold'        # texto en negritas
        )

    plt.tight_layout()
    plt.show()




# --- Visualización de las columnas binarias --- #

fig, axes = plt.subplots(1, len(binary_cols), figsize=(14, 4))
total = len(df)

for ax, col in zip(axes, binary_cols):
    sns.countplot(
        x=df[col],
        ax=ax,
        palette="Set2",
        hue=df[col],
        legend=False,
        order=[0,1]
    )

    ax.set_title(f'{col}')
    ax.set_xlabel("")
    ax.set_ylabel("Cantidad")
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    # Ajustar límite superior para espacio extra
    max_count = df[col].value_counts().max()
    ax.set_ylim(0, max_count * 1.10)

    # Agregar etiquetas de conteo + porcentaje
    for p in ax.patches:
        count = int(p.get_height())
        percent = 100 * count / total
        ax.text(
            p.get_x() + p.get_width()/2,
            p.get_height() + 40,
            f'{percent:.1f}% ({count})',
            ha='center',
            fontweight='bold'
        )

plt.tight_layout()
plt.show()




# --- Buscando correlación entre las variables numéricas y las binarias --- #

# Número de binarias y numéricas
n_bin = len(binary_cols)
n_num = len(numeric_cols)

for num_col in numeric_cols:
    fig, axes = plt.subplots(1, n_bin, figsize=(5*n_bin, 5), sharey=True)

    for i, bin_col in enumerate(binary_cols):
        sns.boxplot(x=df[bin_col], y=df[num_col], ax=axes[i], width=0.5)
        axes[i].set_title(f'{num_col} según {bin_col}')
        axes[i].set_xlabel(bin_col)
        if i == 0:  # solo el primero muestra etiqueta Y
            axes[i].set_ylabel(num_col)
        else:
            axes[i].set_ylabel("")

    plt.suptitle(f"Distribución de {num_col} según variables binarias", 
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.show()




# --- Buscando correlación entre las variables numéricas y categóricas con la variable objetivo --- #

# Definir orden lógico de la variable objetivo
ordered_classes = [
    'insufficient_weight',
    'normal_weight',
    'overweight_level_i',
    'overweight_level_ii',
    'obesity_type_i',
    'obesity_type_ii',
    'obesity_type_iii'
]

# Convertir NObeyesdad a categoría ordenada
cat_type = CategoricalDtype(categories=ordered_classes, ordered=True)
df['NObeyesdad'] = df['NObeyesdad'].astype(cat_type)

# Boxplots para columnas numéricas
n_num = len(numeric_cols)
n_cols = 3
n_rows = (n_num + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(x='NObeyesdad', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} por categoría de NObeyesdad')
    axes[i].tick_params(axis='x', rotation=45)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# Gráficas de barras apiladas para variables categóricas
for col in object_cols_no_target:
    n_categories = df[col].nunique()
    fig_width = (max(2, min(n_categories, 5)) * 2) + 1
    plt.figure(figsize=(fig_width, 4))

    ct = pd.crosstab(df[col], df['NObeyesdad'])
    ct = ct[ordered_classes]
    ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100

    ax = ct_perc.plot(kind='bar', stacked=True, figsize=(fig_width,4))
    plt.title(f'{col} vs NObeyesdad (porcentaje)')
    plt.ylabel('Porcentaje')
    plt.xticks(rotation=45)

    # Mostrar % sobre cada segmento
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.text(x + width/2, y + height/2,
                    f'{height:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')

    # Mover la leyenda fuera del área de la gráfica
    ax.legend(title='NObeyesdad', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass  # main guard added
