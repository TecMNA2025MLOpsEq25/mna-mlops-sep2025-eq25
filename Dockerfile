# ===========================
# Etapa base
# ===========================
FROM python:3.13-slim AS base

# Evita archivos .pyc y usa stdout/stderr sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define el directorio de trabajo
WORKDIR /app

# Copiamos los archivos necesarios para dependencias
COPY requirements.txt .

# Instalamos dependencias del sistema y Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===========================
# Etapa final (runtime)
# ===========================
FROM python:3.13-slim

WORKDIR /app

# Copia desde la etapa base las dependencias ya instaladas
COPY --from=base /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=base /usr/local/bin /usr/local/bin

# Copia el código fuente y el modelo
COPY obesity_estimator/ obesity_estimator/
COPY models/ models/
COPY params.yaml .
COPY dvc.yaml .
COPY README.md .

# Exponer el puerto del servicio
EXPOSE 8000

# Comando de ejecución
CMD ["uvicorn", "obesity_estimator.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
