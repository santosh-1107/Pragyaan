"""Retrieval + severity + end-to-end metrics → eval/results/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, accuracy_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import (  # noqa: E402
    EVAL_DIR,
    EVAL_RESULTS,
    SEVERITY_MODEL_PATH,
    SIM_THRESHOLD,
    TEST_PARQUET,
    TFIDF_PATH,
    TRAIN_PARQUET,
)
from copilot.pipeline import ask  # noqa: E402
from copilot.retrieval import search  # noqa: E402


def eval_severity() -> None:
    if not TEST_PARQUET.exists() or not SEVERITY_MODEL_PATH.exists():
        print("Skip severity eval — missing test data or model")
        return
    df = pd.read_parquet(TEST_PARQUET)
    tfidf = joblib.load(TFIDF_PATH)
    clf = joblib.load(SEVERITY_MODEL_PATH)
    X = tfidf.transform(df["ticket_text_strict"].astype(str))
    y = df["priority"].astype(str)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    macro = f1_score(y, preds, average="macro")
    print(f"Severity TEST — Acc={acc:.4f} macro-F1={macro:.4f}")
    pd.DataFrame([{"accuracy": acc, "macro_f1": macro}]).to_csv(
        EVAL_RESULTS / "severity_test.csv", index=False
    )
    labels = sorted(y.unique())
    cm = confusion_matrix(y, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False)
    ax.set_title("Severity confusion (test)")
    fig.tight_layout()
    fig.savefig(EVAL_RESULTS / "severity_confusion.png", dpi=120)
    plt.close(fig)


def eval_retrieval(k_values=(1, 3, 5), sample_n: int = 300) -> None:
    if not TEST_PARQUET.exists():
        print("Skip retrieval eval — missing test parquet")
        return
    test = pd.read_parquet(TEST_PARQUET).sample(n=min(sample_n, len(pd.read_parquet(TEST_PARQUET))), random_state=42)
    hits = {k: {"queue": 0, "type": 0, "n": 0} for k in k_values}
    for row in test.itertuples():
        results = search(str(row.ticket_text_clean), k=max(k_values))
        for k in k_values:
            top = results[:k]
            hits[k]["n"] += 1
            if any(h.metadata.get("queue") == row.queue for h in top):
                hits[k]["queue"] += 1
            if any(h.metadata.get("type") == row.type for h in top):
                hits[k]["type"] += 1
    rows = []
    for k, h in hits.items():
        n = max(h["n"], 1)
        rows.append(
            {
                "k": k,
                "hit_queue": round(h["queue"] / n, 4),
                "hit_type": round(h["type"] / n, 4),
                "n": h["n"],
            }
        )
    out = pd.DataFrame(rows)
    print("\nRetrieval Hit@k (queue / type):")
    print(out.to_string(index=False))
    out.to_csv(EVAL_RESULTS / "retrieval_hit_at_k.csv", index=False)


def eval_e2e() -> None:
    queries_path = EVAL_DIR / "test_queries.csv"
    if not queries_path.exists():
        print("Skip e2e — missing eval/test_queries.csv")
        return
    qdf = pd.read_csv(queries_path)
    correct_resolve = correct_escalate = missed = 0
    records = []
    for row in qdf.itertuples():
        resp = ask(str(row.question), log=False)
        expected = str(row.expected_route).strip().lower()
        ok = resp.route == expected
        if expected == "resolved" and ok:
            correct_resolve += 1
        elif expected == "escalated" and ok:
            correct_escalate += 1
        else:
            missed += 1
        records.append(
            {
                "question": row.question,
                "expected": expected,
                "got": resp.route,
                "confidence": resp.confidence,
                "ok": ok,
            }
        )
    n = len(qdf)
    summary = {
        "n": n,
        "correct_resolve_pct": round(100 * correct_resolve / n, 1) if n else 0,
        "correct_escalate_pct": round(100 * correct_escalate / n, 1) if n else 0,
        "missed_pct": round(100 * missed / n, 1) if n else 0,
        "threshold": SIM_THRESHOLD,
    }
    print("\nEnd-to-end:")
    print(summary)
    pd.DataFrame([summary]).to_csv(EVAL_RESULTS / "e2e_summary.csv", index=False)
    pd.DataFrame(records).to_csv(EVAL_RESULTS / "e2e_details.csv", index=False)


def main() -> None:
    EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    eval_severity()
    eval_retrieval()
    eval_e2e()
    print(f"\nResults written to {EVAL_RESULTS}")


if __name__ == "__main__":
    main()
