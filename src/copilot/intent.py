"""Intent (ticket type) classifier."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib

from .config import INTENT_MODEL_PATH
from .preprocess import clean_text


@lru_cache(maxsize=1)
def _load() -> Any:
    if not INTENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Intent model missing. Run scripts/train_intent.py first.\n"
            f"Expected: {INTENT_MODEL_PATH}"
        )
    return joblib.load(INTENT_MODEL_PATH)


def predict_intent(question: str) -> str:
    pipe = _load()
    text = clean_text(question, strict=True)
    return str(pipe.predict([text])[0])
