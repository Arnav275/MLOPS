"""
Central configuration for the Student Performance data pipeline.
Loads settings from environment variables (.env file) so credentials
are never hard-coded in source.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", BASE_DIR / "data/raw"))
STAGING_DATA_DIR = Path(os.getenv("STAGING_DATA_DIR", BASE_DIR / "data/staging"))
CLEANED_DATA_DIR = Path(os.getenv("CLEANED_DATA_DIR", BASE_DIR / "data/cleaned"))
ANALYTICAL_DATA_DIR = Path(os.getenv("ANALYTICAL_DATA_DIR", BASE_DIR / "data/analytical"))
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "data/logs"))

for d in [RAW_DATA_DIR, STAGING_DATA_DIR, CLEANED_DATA_DIR, ANALYTICAL_DATA_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "student_analytics")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

PG_CONN_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

UCI_MAT_URL = os.getenv(
    "UCI_MAT_URL",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
)

SOURCE_FILES = {
    "math": "student-mat.csv",
    "portuguese": "student-por.csv",
}

MAX_MARK = 20          # UCI grades G1/G2/G3 are on a 0-20 scale
MAX_ABSENCES_OBSERVED = 93   # Dataset documented max absences, used for attendance % calc
