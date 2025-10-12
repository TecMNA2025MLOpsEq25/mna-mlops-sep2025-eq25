
# ### **Tecnológico de Monterrey**
# 
# #### **Maestría en Inteligencia Artificial Aplicada**
# #### **Clase**: Operaciones de Aprendizaje Automático
# #### **Docentes**: Dr. Gerardo Rodríguez Hernández | Mtro. Ricardo Valdez Hernández | Mtro. Carlos Alberto Vences Sánchez
# 
# ##### **Actividad**: Proyecto: Avance (Fase 1) **Notebook**: Preprocesamiento de datos para análisis y modelado
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



# --- Inicialización --- #

# Librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Configuración inicial
RANDOM_STATE = 42
DATA_PATH = "data/processed/obesity_estimation_clean.csv"
OUTPUT_DIR = "data/prepared"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carga del dataset limpio
df = pd.read_csv(DATA_PATH)
print(f"Dataset de trabajo (df): {df.shape[0]} filas, {df.shape[1]} columnas")
print(df.head())


# ### Preprocesamiento para modelado



# --- Definiciones --- #

# Variable objetivo
target = "NObeyesdad"

# Separar X (features) de y (target)
X = df.drop(columns=[target])
y = df[target]

print("Variable objetivo:", target)
print("Número de variables predictoras:", X.shape[1])




# --- Identificación de tipo de columnas ---#
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print("Numéricas:", num_cols)
print("Categóricas:", cat_cols)




# --- Pipeline de preprocesamiento --- #

# Escalador para variables numéricas
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Codificador para variables categóricas
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combinar transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)




# --- División en conjuntos de entrenamiento, validación y pruebas --- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)




# --- Aplicar transformaciones y generar datasets finales --- #

# Ajustar y transformar
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Obtener nombres de las columnas resultantes
cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
processed_feature_names = num_cols + list(cat_feature_names)

# Crear DataFrames finales
X_train_df = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test.index)

# Añadir la variable objetivo
train_df = pd.concat([X_train_df, y_train], axis=1)
test_df = pd.concat([X_test_df, y_test], axis=1)

print("Train final:", train_df.shape)
print("Test final:", test_df.shape)




# --- Guardar datasets procesados y el preprocesador, para versionado --- #

# Rutas de conjuntos y preprocesador
train_path = os.path.join(OUTPUT_DIR, "train_prepared.csv")
test_path = os.path.join(OUTPUT_DIR, "test_prepared.csv")
preprocessor_path = os.path.join(OUTPUT_DIR, "preprocessor.pkl")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
joblib.dump(preprocessor, preprocessor_path)

print("Archivos guardados:")
print("-", train_path)
print("-", test_path)
print("-", preprocessor_path)




# --- Revisión de las proporciones de la variable objetivo en ambos conjuntos --- #

# Lista del orden deseado
ordered_classes = [
    'insufficient_weight',
    'normal_weight',
    'overweight_level_i',
    'overweight_level_ii',
    'obesity_type_i',
    'obesity_type_ii',
    'obesity_type_iii'
]

# Calcular proporciones
train_dist = y_train.value_counts(normalize=True).round(3)
test_dist = y_test.value_counts(normalize=True).round(3)

# Crear DataFrame combinando train y test
dist_df = pd.DataFrame({
    'Train': train_dist,
    'Test': test_dist
})

# Reordenar según el orden lógico
dist_df = dist_df.reindex(ordered_classes)

# Mostrar tabla
dist_df.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"))


if __name__ == '__main__':
    pass  # main guard added
