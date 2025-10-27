from pathlib import Path

# Datos crudos y procesados
RAW_FILEPATH = Path("data/raw/obesity_estimation_modified.csv")
PROCESSED_FILEPATH = Path("data/processed/obesity_estimation_clean.csv")

# Intermedios
INTERIM_DIR = Path("data/interim")
TRAIN_PREPARED = INTERIM_DIR / "train_prepared.csv"
TEST_PREPARED  = INTERIM_DIR / "test_prepared.csv"

# Modelos y reportes
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
