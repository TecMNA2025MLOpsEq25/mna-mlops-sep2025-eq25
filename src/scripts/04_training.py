
# ### **Tecnológico de Monterrey**
# 
# #### **Maestría en Inteligencia Artificial Aplicada**
# #### **Clase**: Operaciones de Aprendizaje Automático
# #### **Docentes**: Dr. Gerardo Rodríguez Hernández | Mtro. Ricardo Valdez Hernández | Mtro. Carlos Alberto Vences Sánchez
# 
# ##### **Actividad**: Proyecto: Avance (Fase 1) **Notebook**: Modelo de aprendizaje automático
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
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, matthews_corrcoef, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import joblib
import mlflow
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings


# Configuración inicial
DATA_DIR = "data/prepared"
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "Obesity_Classification"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Remoción de advertencias
os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"
warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty is 'elasticnet'", category=UserWarning)
warnings.filterwarnings("ignore", message="The max_iter was reached which means the coef_ did not converge")

# Carga del dataset
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_prepared.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_prepared.csv"))

X_train = train_df.drop(columns=['NObeyesdad'])
y_train = train_df['NObeyesdad']
X_test = test_df.drop(columns=['NObeyesdad'])
y_test = test_df['NObeyesdad']

print("Datos cargados correctamente:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")




# --- Código utilitario --- #

# Función para calcular métricas de evaluación de modelos
def evaluate_model(y_true, y_pred, y_proba=None, average="macro"):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average=average, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }

    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except ValueError:
            metrics["log_loss"] = np.nan

        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except Exception:
            metrics["roc_auc_ovr"] = np.nan

    return metrics


# Función auxiliar para ejecución de modelos en MLflow
def run_experiment(model, model_name, X_train, X_test, y_train, y_test, params=None):

    with mlflow.start_run(run_name=model_name):
        # Log de parámetros
        mlflow.log_param("model_name", model_name)
        if params:
            mlflow.log_params(params)

        # Entrenamiento
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Evaluación
        metrics = evaluate_model(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        # Registro del modelo
        input_example = X_train.head(1)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            name=model_name,
            input_example=input_example,
            signature=signature
        )

        print(f"\nResultados de {model_name}:")
        for k, v in metrics.items():
            print(f"{k:20s}: {v:.4f}")

    return metrics


# ### Modelado



# --- Modelo base y ejecución inicial --- #

# Modelos a probar
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500, solver="saga", random_state=42),
    "SVC": SVC(probability=True, random_state=42)
}

results = []
for name, model in models.items():
    metrics = run_experiment(model, name, X_train, X_test, y_train, y_test)
    results.append({"Modelo": name, **metrics})

results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
print(results_df)
results_df.to_csv("reports/model_comparison.csv", index=False)




# --- Ajuste de Hiperparámetros --- #

# Grids con valores a usar
param_grids = {
    "RandomForest": {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2", "elasticnet"],
        "l1_ratio": [0, 0.5, 1]
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }
}

best_models = {}
scoring_metric = "f1_macro"

for name, model in models.items():
    print(f"\nAjustando {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring=scoring_metric, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_
    print(f"Mejores parámetros: {grid.best_params_}")
    print(f"Mejor {scoring_metric}: {grid.best_score_:.4f}")

    input_example = X_train.head(1)
    signature = infer_signature(X_train, model.predict(X_train))

    # Registrar en MLflow
    with mlflow.start_run(run_name=f"{name}_tuned"):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric(f"best_cv_{scoring_metric}", grid.best_score_)
        mlflow.sklearn.log_model(grid.best_estimator_, name="model", input_example=input_example, signature=signature)




# --- Guardar el mejor modelo --- #

final_results = []

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    metrics = evaluate_model(y_test, y_pred, y_proba)
    final_results.append({"Modelo": name, **metrics})

final_df = pd.DataFrame(final_results).sort_values(by="f1_macro", ascending=False)
print(final_df)
final_df.to_csv("reports/final_model_comparison.csv", index=False)

best_model_name = final_df.iloc[0]["Modelo"]
best_model = best_models[best_model_name]
print(f"\nMejor modelo final: {best_model_name}")

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, f"models/best_model.pkl")
print(f"Modelo guardado en: /models/best_model.pkl")

# Matriz de confusión
cm = confusion_matrix(y_test, best_model.predict(X_test))
cm_df = pd.DataFrame(cm, index=best_model.classes_, columns=best_model.classes_)
print("\nMatriz de Confusión:")
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusión - {best_model_name}")
plt.ylabel("Real")
plt.xlabel("Predicho")

confusion_path = "reports/confusion_matrix.png"
plt.savefig(confusion_path)
plt.close()

print(f"Matriz de confusión guardada en: {confusion_path}")
cm_df.to_csv("reports/confusion_matrix.csv")


if __name__ == '__main__':
    pass  # main guard added
