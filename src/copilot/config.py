"""Environment + thresholds for the Onboarding Knowledge Copilot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SIM_THRESHOLD: float = float(os.getenv("SIM_THRESHOLD", "0.55"))
CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma")))
SQLITE_PATH: Path = Path(os.getenv("SQLITE_PATH", str(ROOT / "data" / "copilot.db")))
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TOP_K: int = int(os.getenv("TOP_K", "5"))

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "eval"
EVAL_RESULTS = EVAL_DIR / "results"

TICKETS_CSV = DATA_RAW / "dataset-tickets-multi-lang-4-20k.csv"
TICKETS_PARQUET = DATA_PROCESSED / "tickets_en.parquet"
TRAIN_PARQUET = DATA_PROCESSED / "train.parquet"
VAL_PARQUET = DATA_PROCESSED / "val.parquet"
TEST_PARQUET = DATA_PROCESSED / "test.parquet"

SEVERITY_MODEL_PATH = MODELS_DIR / "severity.joblib"
INTENT_MODEL_PATH = MODELS_DIR / "intent.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf.joblib"
INTENT_TFIDF_PATH = MODELS_DIR / "intent_tfidf.joblib"

COLLECTION_NAME = "tickets_kb"

# Severity routing thresholds (score = P(medium)*0.5 + P(high)*1.0)
URGENT_THRESHOLD = 0.70
NORMAL_THRESHOLD = 0.40
