# Proyecto MLOps — Clasificación del Nivel de Obesidad

Repositorio oficial del proyecto de clasificación de obesidad utilizando técnicas de Machine Learning y principios de MLOps.  
Desarrollado por **Equipo 25 - MNA 2025, Tecnológico de Monterrey**.

Repositorio: [TecMNA2025MLOpsEq25/mna-mlops-sep2025-eq25](https://github.com/TecMNA2025MLOpsEq25/mna-mlops-sep2025-eq25/tree/master)

---

## 1. Propósito y contexto

El objetivo del proyecto es construir un pipeline reproducible de aprendizaje automático que permita clasificar el nivel de obesidad de una persona con base en sus hábitos alimenticios, actividad física y características antropométricas.

El flujo completo está diseñado bajo prácticas de MLOps, utilizando DVC (Data Version Control) para controlar dependencias, versionar datos, registrar métricas y reproducir los experimentos con un solo comando (`dvc repro`).

El proyecto se alinea con buenas prácticas de ingeniería de datos y aprendizaje de máquina:
- Limpieza y preparación sistemática de datos.
- Modularización por etapas (`prepare`, `eda`, `preprocessing`, `training`, `evaluation`, `model_plots`).
- Evaluación automática de modelos y trazabilidad de métricas.
- Visualización e interpretación final de los resultados.

---

## 2. Estructura general del pipeline (DVC)

El pipeline se encuentra definido en el archivo [`dvc.yaml`](https://github.com/TecMNA2025MLOpsEq25/mna-mlops-sep2025-eq25/blob/master/dvc.yaml):

| Etapa | Script | Descripción | Entradas | Salidas |
|-------|---------|-------------|-----------|----------|
| **prepare** | `obesity_estimator/dataset.py` | Limpieza de datos, eliminación de duplicados y outliers. | `data/raw/obesity_estimation_modified.csv` | `data/processed/obesity_estimation_clean.csv` |
| **eda** | `obesity_estimator/plots.py` | Análisis exploratorio de datos (EDA): distribuciones, correlaciones y visualizaciones. | Datos procesados | `reports/figures/eda/*` |
| **preprocessing** | `obesity_estimator/features.py` | Codificación de variables, escalado, división Train/Test y guardado del pipeline. | Datos procesados | `data/interim/*`, `preprocessor.pkl` |
| **training** | `obesity_estimator/modeling/train.py` | Entrenamiento y ajuste de hiperparámetros con GridSearchCV y RandomizedSearchCV. | Datos preprocesados | `models/*.joblib`, `reports/metrics.json` |
| **evaluation** | `obesity_estimator/modeling/predict.py` | Evaluación y generación de métricas finales, matriz de confusión y comparativa. | Modelos entrenados | `reports/evaluation_results.csv`, `reports/confusion_matrix.png` |
| **model_plots** | `obesity_estimator/modeling/plot_curves.py` | Curvas ROC y PR para cada modelo y comparativas globales. | Modelos + test | `reports/figures/models/*`, `reports/roc_auc_by_model.csv`, `reports/pr_auc_by_model.csv` |

---

## 3. Fases del pipeline: descripción técnica y análisis de resultados

### Fase 1: prepare — Limpieza y validación de datos
**Objetivo:** garantizar la integridad del conjunto de datos y eliminar registros duplicados o inconsistentes.

- Operaciones realizadas:
  - Carga del archivo original `data/raw/obesity_estimation_modified.csv`.
  - Eliminación de 24 registros duplicados.
  - Validación de tipos de datos y valores atípicos.
  - Exportación del dataset limpio a `data/processed/obesity_estimation_clean.csv`.

**Resultados clave:**
- Tamaño inicial: 2,111 filas → Final: 2,087 filas.
- Sin pérdida de variables críticas.
- Balance conservado entre clases de “NObeyesdad”.

**Conclusión:** el dataset quedó consistente para el modelado sin introducir sesgo o pérdida de información.

---

### Fase 2: eda — Exploratory Data Analysis
**Objetivo:** comprender la distribución, relaciones y patrones entre las variables.

- Acciones principales:
  - Generación automática de histogramas y conteos de categorías.
  - Análisis de correlaciones numéricas con la variable objetivo.
  - Cálculo de estadísticas descriptivas (`numeric_summary.csv`, `categorical_summary.csv`).
  - Visualización de relaciones con la variable target (`num_vs_target.png`, `cat_vs_target_*.png`).

**Hallazgos importantes:**
- Edad y peso muestran correlaciones positivas con el tipo de obesidad.
- Variables conductuales como FAF (actividad física) y FAVC (consumo de comida calórica) influyen significativamente.
- Las clases de género y medio de transporte (MTRANS) presentan diferencias notables.

**Ejemplos de visualizaciones:**
![Distribución Peso](https://github.com/TecMNA2025MLOpsEq25/mna-mlops-sep2025-eq25/blob/master/reports/figures/eda/dist_Weight.png)
![Conteo por género](https://github.com/TecMNA2025MLOpsEq25/mna-mlops-sep2025-eq25/blob/master/reports/figures/eda/cat_count_Gender.png)

**Conclusión:** el EDA permitió identificar los predictores más influyentes y validar la diversidad de clases, asegurando un modelo robusto.

---

### Fase 3: preprocessing — Codificación, escalado y división de datos
**Objetivo:** transformar los datos para hacerlos compatibles con los modelos de aprendizaje automático.

- Tareas implementadas:
  - Codificación One-Hot de variables categóricas (Gender, CAEC, CALC, MTRANS).
  - Normalización de variables numéricas.
  - División estratificada en conjuntos train (70%) y test (30%).
  - Serialización del preprocesador (preprocessor.pkl).

**Resultados:**
- 24 columnas finales tras la transformación.
- train_prepared.csv (1460 filas) y test_prepared.csv (627 filas).

**Conclusión:** la preparación mantuvo la proporción de clases y estandarizó los valores, optimizando el rendimiento de los modelos.

---

### Fase 4: training — Entrenamiento y ajuste de hiperparámetros
**Objetivo:** encontrar el modelo con mejor capacidad de generalización mediante búsqueda de hiperparámetros.

**Modelos evaluados:**
1. Logistic Regression — baseline, regularización L2.  
2. Random Forest — optimización de profundidad y número de árboles.  
3. HistGradientBoosting — ajuste de learning rate y número de iteraciones.  
4. SVC (RBF) — búsqueda sobre C y gamma.

**Método de ajuste:**
- GridSearchCV si combinaciones ≤ 250.
- RandomizedSearchCV si combinaciones > 250.
- Validación cruzada k=5, métrica principal: F1-macro.

**Resultados:**
| Modelo | F1-macro | ROC-AUC | Observación |
|---------|-----------|----------|--------------|
| HistGradientBoosting | 0.9673 | 0.972 | Mejor equilibrio entre precisión y recall, sin sobreajuste. |
| Random Forest | 0.952 | 0.962 | Ligeramente más variable, buena estabilidad. |
| SVC (RBF) | 0.935 | 0.941 | Precisión alta, sensibilidad menor. |
| Logistic Regression | 0.918 | 0.924 | Modelo base, mayor sesgo. |

**Conclusión:**  
El modelo HistGradientBoosting obtuvo la mejor métrica F1 y AUC, mostrando excelente generalización. No se observa sobreentrenamiento, ya que las métricas de validación y prueba son consistentes.

---

### Fase 5: evaluation — Evaluación y métricas finales
**Objetivo:** comparar los modelos entrenados y analizar el desempeño del mejor clasificador.

**Resultados generados:**
- evaluation_results.csv: contiene precisión, recall y F1 por modelo.
- confusion_matrix.png: matriz visual de desempeño global.

![Matriz de confusión](https://github.com/TecMNA2025MLOpsEq25/mna-mlops-sep2025-eq25/blob/master/reports/confusion_matrix.png)

**Interpretación:**
- Excelente rendimiento en clases extremas como Obesity Type III e Insufficient Weight.
- Las confusiones se concentran en las clases intermedias (Overweight I/II).
- El recall superior al 0.96 confirma buena cobertura de todas las clases.

---

### Fase 6: model_plots — Curvas ROC y PR
**Objetivo:** analizar visualmente la discriminación y precisión de cada modelo.

**Archivos generados:**
- reports/figures/models/roc_macro_compare.png
- reports/figures/models/pr_macro_compare.png
- reports/roc_auc_by_model.csv
- reports/pr_auc_by_model.csv

**Resultados visuales:**
![ROC comparativa](https://github.com/TecMNA2025MLOpsEq25/mna-mlops-sep2025-eq25/blob/master/reports/figures/models/roc_macro_compare.png)
![PR comparativa](https://github.com/TecMNA2025MLOpsEq25/mna-mlops-sep2025-eq25/blob/master/reports/figures/models/pr_macro_compare.png)

**Conclusión:**  
El modelo HistGradientBoosting domina tanto en AUC-ROC (0.972) como en AUC-PR (0.970), confirmando su robustez.

---

## 4. Ejecución del pipeline completo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc repro
```

---

## 5. Evaluación del desempeño general

| Modelo | F1-macro | Accuracy | ROC-AUC | PR-AUC | Observación |
|---------|-----------|----------|----------|----------|--------------|
| HistGradientBoosting | 0.9673 | 0.969 | 0.972 | 0.970 | Modelo final, sin sobreentrenamiento. |
| Random Forest | 0.952 | 0.958 | 0.962 | 0.957 | Buen recall, ligera varianza. |
| SVC (RBF) | 0.935 | 0.942 | 0.941 | 0.940 | Precisión alta, menor recall. |
| Logistic Regression | 0.918 | 0.921 | 0.924 | 0.926 | Baseline. |

---

## 6. Conclusiones generales

1. DVC garantiza la trazabilidad completa de los datos, métricas y modelos.  
2. HistGradientBoosting ofrece la mejor relación entre precisión y recall.  
3. El análisis EDA permitió comprender el peso relativo de cada variable.  
4. La modularidad del código permite extender nuevas fuentes de datos y modelos.  
5. El pipeline es reproducible y escalable para despliegue en entornos productivos.

---

## 7. Próximos pasos

- Integrar FastAPI para servir inferencias.  
- Configurar CI/CD con GitHub Actions + DVC.  
- Extender monitoreo con MLflow Tracking.  
- Desplegar el modelo en un entorno reproducible (Docker/Kubernetes).

