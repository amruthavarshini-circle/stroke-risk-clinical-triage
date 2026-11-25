from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "stroke_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"

TARGET = "stroke"
ID_COLUMN = "id"

NUMERIC_FEATURES = [
    "age", "hypertension", "heart_disease",
    "avg_glucose_level", "bmi"
]
CATEGORICAL_FEATURES = [
    "gender", "ever_married", "work_type",
    "Residence_type", "smoking_status"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
