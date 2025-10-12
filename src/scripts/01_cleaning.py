
# ### **Tecnológico de Monterrey**
# 
# #### **Maestría en Inteligencia Artificial Aplicada**
# #### **Clase**: Operaciones de Aprendizaje Automático
# #### **Docentes**: Dr. Gerardo Rodríguez Hernández | Mtro. Ricardo Valdez Hernández | Mtro. Carlos Alberto Vences Sánchez
# 
# ##### **Actividad**: Proyecto: Avance (Fase 1) - **Notebook**: Limpieza y corrección de datos
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

import pandas as pd
import numpy as np




# --- Cargar Dataset --- #

df = pd.read_csv('data/raw/obesity_estimation_modified.csv')
print('Dataset de trabajo (df)', df.shape)


# ### Exploración inicial
# Revisar información general, tipos de datos, primeros registros y estadísticas descriptivas.



# --- Revisión inicial --- #

print(df.head())




# Información general y tipos
print(df.info())


# ### Correcciones
# Removemos columnas sin valor, corregimos datos en columnas, removemos valores atípicos obvios, imputamos valores faltantes.



# --- Remover columna mixed_type_col ---#

# Justificación:
# - No tiene valores uniformes
# - No parece guardar información que sea valiosa

df.drop(columns=['mixed_type_col'], axis=1, inplace=True)

# Confirmamos que la columna fue removida
print(df.head())




# --- Corrección de tipos de datos --- #

# Justificación:
# Todas las columnas son identificadas como objeto, por lo cual es necesario revisar y corregir
# los tipos de datos, para obtener datos adecuados en cada columna y poder actuar en ellos.

# Según la página base del dataset, tenemos valores:
# - Categóricos: Gender, CAEC, CALC, MTRANS, NObeyesdad
# - Enteros: FCVC, TUE
# - Flotantes: Age, Height, Weight, NCP, CH2O, FAF
# - Binarios: family_history_with_overweight, FAVC, SMOKE, SCC

# Sin embargo, de acuerdo a la exploración visual encontramos valores flotantes en FCVC y TUE, así que los consideraremos flotantes.
numeric_cols = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']

# Inspeccionemos los primeros 20 valores de cada columna numérica
print("Valores en columnas numéricas:\n")
for col in numeric_cols:
    print(f"{col}: {df[col].unique()[:20]}")

# Columnas binarias (yes, no)
binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

# Inspeccionemos los primeros 20 valores de cada columna binaria
print("\nValores en columnas binarias:\n")
for col in binary_cols:
    print(f"{col}: {df[col].unique()[:20]}")

# Columnas de tipo texto
object_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad']

# Inspeccionemos los primeros 20 valores de cada columna de texto
print("\nValores en columnas de texto:\n")
for col in object_cols:
    print(f"{col}: {df[col].unique()[:20]}")




# --- Corrección y limpieza --- #

# Justificación: encontramos algunas cosas irregulares, por ejemplo:
# - Espacios en blanco alreadedor del valor numérico: ' 3.0 '
# - Valores textuales: 'invalid' o ' NAN ' o '?'
# - Encontramos diferentes cadenas con diferente construcción de mayúsculas y minúsculas

# Convertir todas las columnas numéricas a float
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convertimos las columnas que identificamos como binarias a valores binarios (0 y 1)
binary_map = {'yes': 1, 'no': 0}
for col in binary_cols:
    df[col] = (
        df[col]
        .astype(str)              # Convertir todo a string
        .str.strip()              # Eliminar espacios
        .str.lower()              # Uniformar a minúsculas
        .replace({'nan': np.nan}) # Convertir texto "nan"/" NAN " a NaN real
        .map(binary_map)          # Mapear yes/no a 1/0
    )

# Convertir a enteros con soporte para NaN
df[binary_cols] = df[binary_cols].astype('Int64')

# Aplicamos strip y lower en todas las columnas de texto
for col in object_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Reemplazar valores no numéricos por NaN
df.replace({'nan': np.nan, '?': np.nan, 'error': np.nan, 'invalid': np.nan, 'n/a': np.nan, 'null': np.nan}, inplace=True)

# Verificar resultados
print(df.info())
print(df.head())




# --- Validación visual ---#

# Reimprimimos los valores de muestra una vez corregidos
print("Valores en columnas numéricas:\n")
for col in numeric_cols:
    print(f"{col}: {df[col].unique()[:20]}")

# Inspeccionemos los primeros 20 valores de cada columna binaria
print("\nValores en columnas binarias:\n")
for col in binary_cols:
    print(f"{col}: {df[col].unique()[:20]}")

# Inspeccionemos los primeros 20 valores de cada columna de texto
print("\nValores en columnas de texto:\n")
for col in object_cols:
    print(f"{col}: {df[col].unique()[:20]}")




# --- Filtrado de outliers obvios --- #

# Justificación:
# - Hay valores muy fuera de los esperados o simplemente inválidos

# Definir rangos normales para cada columna numérica
valid_ranges = {
    'Age': (5, 120),         # años
    'Height': (1.3, 2.2),    # metros
    'Weight': (30, 200),     # kg
    'FCVC': (1, 3),          # frecuencia comida principal
    'NCP': (1, 3),           # número de comidas
    'CH2O': (1, 3),          # litros de agua
    'FAF': (0, 5),           # actividad física
    'TUE': (0, 3)            # tiempo frente a pantalla
}

# Aplicar filtros: si el valor cae fuera del rango establecido como normal, poner NaN
for col, (min_val, max_val) in valid_ranges.items():
    df.loc[(df[col] < min_val) | (df[col] > max_val), col] = np.nan

# Verificar resultados
for col in valid_ranges.keys():
    print(f"{col}: valores únicos (primeros 20) después de filtrar outliers")
    print(df[col].unique()[:20])
    print("----")




# --- Imputación de NaN y limpieza final --- #

# Columnas numéricas: imputar con mediana
for col in numeric_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    print(f"Columna {col}: mediana imputada = {median_value}")

# Columnas categóricas: imputar con moda
for col in object_cols:
    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value)
    print(f"Columna {col}: modo imputado = '{mode_value}'")

# Columnas binarias: imputar con moda
for col in binary_cols:
    moda = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(moda)

# Buscar registros duplicados, es decir, que sean exactamente iguales
dups = df.duplicated().sum()
print(f'Registros duplicados: {dups}\n')

# En caso de encontrar alguno, eliminarlos
if dups > 0:
    print('Eliminando duplicados ...')
    df = df.drop_duplicates()
    dups = df.duplicated().sum()
    print(f'Registros duplicados: {dups}\n')

# Veamos las nuevas dimensiones del dataset
print('Nuevas dimensiones del dataset (df)', df.shape)

# Verificación final
print("\nInformación final del dataset:")
print(df.info())
print(df.head())




# --- Guardar versión limpia después del procesamiento realizado --- #

# Generamos una copia del dataframe (df_clean) para continuar en ella el EDA
df_clean = df.copy()

# Y versionamos nuestra copia limpia
df_clean.to_csv('data/processed/obesity_estimation_clean.csv', index=False)


# ### Inspección Visual
# Revisemos las estadísticas por tipo de columna y el conteo de valores nulos por columna.



# --- Estadísticas descriptivas --- #

# Columnas numéricas
desc_num = df_clean[numeric_cols].describe()
print(desc_num)

# Columnas binarias (estadísticas tipo object)
desc_bin = df_clean[binary_cols].astype('object').describe()
print(desc_bin)

# Columnas de texto
desc_cat = df_clean[object_cols].describe()
print(desc_cat)




# Conteo de nulos por columna
df_clean.isnull().sum()


if __name__ == '__main__':
    pass  # main guard added
