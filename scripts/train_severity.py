"""Train + compare severity models; save best + TF-IDF."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import (  # noqa: E402
    EVAL_RESULTS,
    MODELS_DIR,
    SEVERITY_MODEL_PATH,
    TFIDF_PATH,
    TRAIN_PARQUET,
    VAL_PARQUET,
)


def load_xy(path: Path):
    df = pd.read_parquet(path)
    X = df["ticket_text_strict"].astype(str).tolist()
    y = df["priority"].astype(str).tolist()
    return X, y


def eval_model(name: str, clf, X_train, y_train, X_val, y_val, fit_X=None):
    """fit_X overrides X_train for embedding-based models."""
    train_in = fit_X if fit_X is not None else X_train
    t0 = time.perf_counter()
    clf.fit(train_in, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = clf.predict(X_val if fit_X is None else X_val)
    infer_s = (time.perf_counter() - t1) / max(len(y_val), 1) * 1000  # ms/sample

    acc = accuracy_score(y_val, preds)
    macro = f1_score(y_val, preds, average="macro")
    print(f"\n=== {name} ===")
    print(f"Accuracy={acc:.4f}  macro-F1={macro:.4f}  train={train_s:.1f}s  infer={infer_s:.3f}ms/sample")
    print(classification_report(y_val, preds, digits=3))
    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro,
        "train_s": train_s,
        "infer_ms": infer_s,
        "clf": clf,
        "preds": preds,
    }


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    from copilot.config import EMBEDDING_MODEL

    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_xy(TRAIN_PARQUET)
    X_val, y_val = load_xy(VAL_PARQUET)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2000,
        stop_words="english",
        min_df=2,
    )
    Xtr = tfidf.fit_transform(X_train)
    Xva = tfidf.transform(X_val)

    results = []

    # 1) Logistic Regression (balanced)
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    results.append(eval_model("LogReg (TF-IDF)", lr, Xtr, y_train, Xva, y_val))

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=30,
    )
    results.append(eval_model("RandomForest (TF-IDF)", rf, Xtr, y_train, Xva, y_val))

    # 3) MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=30,
        random_state=42,
        early_stopping=True,
    )
    results.append(eval_model("MLP (TF-IDF)", mlp, Xtr, y_train, Xva, y_val))

    # 4) MiniLM embeddings + LogReg
    print("\nEncoding embeddings for MiniLM+LogReg…")
    Emb_tr = embed_texts(X_train)
    Emb_va = embed_texts(X_val)
    lr_emb = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    results.append(
        eval_model("MiniLM+LogReg", lr_emb, Emb_tr, y_train, Emb_va, y_val, fit_X=Emb_tr)
    )

    # Comparison table
    table = pd.DataFrame(
        [
            {
                "model": r["model"],
                "accuracy": round(r["accuracy"], 4),
                "macro_f1": round(r["macro_f1"], 4),
                "train_s": round(r["train_s"], 2),
                "infer_ms": round(r["infer_ms"], 3),
            }
            for r in results
        ]
    )
    print("\n===== SEVERITY MODEL COMPARISON =====")
    print(table.to_string(index=False))
    table.to_csv(EVAL_RESULTS / "severity_comparison.csv", index=False)

    # Prefer TF-IDF models for top_terms explainability; pick best by macro-F1 among TF-IDF
    tfidf_results = [r for r in results if "TF-IDF" in r["model"]]
    best = max(tfidf_results, key=lambda r: r["macro_f1"])
    overall_best = max(results, key=lambda r: r["macro_f1"])
    print(f"\nBest overall: {overall_best['model']} (macro-F1={overall_best['macro_f1']:.4f})")
    print(f"Saving TF-IDF best for explainability: {best['model']}")

    # Save as a sklearn Pipeline for easy loading
    pipe = Pipeline([("tfidf", tfidf), ("clf", best["clf"])])
    # Re-fit pipeline end-to-end so tfidf+clf are consistent
    # (clf already fitted on same tfidf transform — wrap carefully)
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(best["clf"], SEVERITY_MODEL_PATH)
    joblib.dump(
        {"model_name": best["model"], "classes": list(best["clf"].classes_)},
        MODELS_DIR / "severity_meta.joblib",
    )
    # Also keep a ready pipeline object
    joblib.dump(pipe, MODELS_DIR / "severity_pipeline.joblib")
    print(f"Saved to {SEVERITY_MODEL_PATH}, {TFIDF_PATH}")


if __name__ == "__main__":
    main()
