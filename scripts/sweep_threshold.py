"""Sweep SIM_THRESHOLD on in-KB vs novel/out-of-scope queries to evaluate trade-offs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import EVAL_DIR, EVAL_RESULTS
from copilot.retrieval import search


def main() -> None:
    EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    queries_path = EVAL_DIR / "test_queries.csv"
    if not queries_path.exists():
        print(f"Missing {queries_path}")
        return

    qdf = pd.read_csv(queries_path)
    print(f"Loaded {len(qdf)} test queries for threshold sweep.")

    # Pre-compute top similarity for each query
    records = []
    for row in qdf.itertuples():
        hits = search(str(row.question), k=1)
        sim = hits[0].similarity if hits else 0.0
        records.append(
            {
                "question": row.question,
                "expected": str(row.expected_route).strip().lower(),
                "top_similarity": sim,
            }
        )
    df = pd.DataFrame(records)

    in_kb = df[df["expected"] == "resolved"]
    novel = df[df["expected"] == "escalated"]

    thresholds = [round(t * 0.05, 2) for t in range(6, 18)]  # 0.30 to 0.85
    sweep_rows = []

    for t in thresholds:
        # If sim >= t, predicted route is 'resolved'; else 'escalated'
        # True Positives: in_kb with sim >= t (auto-resolved)
        tp = (in_kb["top_similarity"] >= t).sum()
        # False Negatives: in_kb with sim < t (escalated when it could be resolved)
        fn = (in_kb["top_similarity"] < t).sum()
        # False Positives: novel with sim >= t (falsely resolved!)
        fp = (novel["top_similarity"] >= t).sum()
        # True Negatives: novel with sim < t (correctly escalated)
        tn = (novel["top_similarity"] < t).sum()

        auto_resolve_rate = tp / len(in_kb) if len(in_kb) else 0.0
        false_resolve_rate = fp / len(novel) if len(novel) else 0.0
        correct_escalate_rate = tn / len(novel) if len(novel) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = auto_resolve_rate
        accuracy = (tp + tn) / len(df) if len(df) else 0.0

        sweep_rows.append(
            {
                "threshold": t,
                "auto_resolve_pct": round(auto_resolve_rate * 100, 1),
                "false_resolve_pct": round(false_resolve_rate * 100, 1),
                "correct_escalate_pct": round(correct_escalate_rate * 100, 1),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "accuracy": round(accuracy * 100, 1),
            }
        )

    sweep_df = pd.DataFrame(sweep_rows)
    print("\nThreshold sweep results:")
    print(sweep_df.to_string(index=False))

    out_csv = EVAL_RESULTS / "threshold_sweep.csv"
    sweep_df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV to {out_csv}")

    # Plot trade-off curves
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        sweep_df["threshold"],
        sweep_df["auto_resolve_pct"],
        marker="o",
        color="#2b5c8f",
        label="Auto-Resolve Rate (Recall on In-KB %)",
    )
    ax1.plot(
        sweep_df["threshold"],
        sweep_df["false_resolve_pct"],
        marker="s",
        color="#d9534f",
        label="False Resolve Rate (% False Alarm)",
    )
    ax1.axvline(0.55, color="gray", linestyle="--", label="Current Threshold (0.55)")
    ax1.set_xlabel("Similarity Threshold")
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Similarity Threshold Sweep: Auto-Resolve vs False-Resolve")
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    out_png = EVAL_RESULTS / "threshold_sweep.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()

