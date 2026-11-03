"""Train intent (ticket type) classifier."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import (  # noqa: E402
    EVAL_RESULTS,
    INTENT_MODEL_PATH,
    INTENT_TFIDF_PATH,
    MODELS_DIR,
    TRAIN_PARQUET,
    VAL_PARQUET,
)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_PARQUET)
    val = pd.read_parquet(VAL_PARQUET)
    X_train = train["ticket_text_strict"].astype(str)
    y_train = train["type"].astype(str)
    X_val = val["ticket_text_strict"].astype(str)
    y_val = val["type"].astype(str)

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=2000,
                    stop_words="english",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    preds = pipe.predict(X_val)
    acc = accuracy_score(y_val, preds)
    macro = f1_score(y_val, preds, average="macro")

    print(f"Intent LogReg — Accuracy={acc:.4f}  macro-F1={macro:.4f}  train={train_s:.1f}s")
    print(classification_report(y_val, preds, digits=3))

    pd.DataFrame(
        [{"model": "LogReg (TF-IDF)", "accuracy": acc, "macro_f1": macro, "train_s": train_s}]
    ).to_csv(EVAL_RESULTS / "intent_comparison.csv", index=False)

    joblib.dump(pipe, INTENT_MODEL_PATH)
    joblib.dump(pipe.named_steps["tfidf"], INTENT_TFIDF_PATH)
    print(f"Saved to {INTENT_MODEL_PATH}")


if __name__ == "__main__":
    main()
