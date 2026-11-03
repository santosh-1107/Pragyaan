"""Severity model: load, score, top contributing TF-IDF terms."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from .config import MODELS_DIR, SEVERITY_MODEL_PATH, TFIDF_PATH, URGENT_THRESHOLD, NORMAL_THRESHOLD
from .preprocess import clean_text
from .schemas import SeverityInfo


@lru_cache(maxsize=1)
def _load() -> tuple[Any, Any]:
    if not SEVERITY_MODEL_PATH.exists() or not TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"Severity model missing. Run scripts/train_severity.py first.\n"
            f"Expected: {SEVERITY_MODEL_PATH}"
        )
    tfidf = joblib.load(TFIDF_PATH)
    clf = joblib.load(SEVERITY_MODEL_PATH)
    return tfidf, clf


def routing_bucket(score: float) -> str:
    if score >= URGENT_THRESHOLD:
        return "Urgent"
    if score >= NORMAL_THRESHOLD:
        return "Normal"
    return "Backlog"


def _top_terms(tfidf, clf, text: str, predicted: str, k: int = 5) -> list[str]:
    vec = tfidf.transform([text])
    feature_names = np.array(tfidf.get_feature_names_out())
    row = vec.toarray()[0]
    nonzero = np.where(row > 0)[0]
    if len(nonzero) == 0:
        return []

    if hasattr(clf, "coef_"):
        classes = list(clf.classes_)
        try:
            class_idx = classes.index(predicted)
        except ValueError:
            return []
        contributions = clf.coef_[class_idx] * row
    elif hasattr(clf, "feature_importances_"):
        # RandomForest / tree models: importance * presence in this doc
        contributions = clf.feature_importances_ * row
    else:
        return []

    ranked = nonzero[np.argsort(contributions[nonzero])[::-1]]
    terms = [str(feature_names[i]) for i in ranked[:k] if contributions[i] > 0]
    return terms


def predict_severity(question: str) -> SeverityInfo:
    tfidf, clf = _load()
    text = clean_text(question, strict=True)
    X = tfidf.transform([text])
    label = str(clf.predict(X)[0])

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        classes = list(clf.classes_)
        pmap = {c: float(p) for c, p in zip(classes, proba)}
    else:
        pmap = {label: 1.0}

    score = pmap.get("medium", 0.0) * 0.5 + pmap.get("high", 0.0) * 1.0
    terms = _top_terms(tfidf, clf, text, label, k=5)
    return SeverityInfo(label=label, score=round(score, 4), top_terms=terms)  # type: ignore[arg-type]
