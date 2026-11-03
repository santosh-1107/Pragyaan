"""Embed train-split tickets into persistent ChromaDB."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import CHROMA_DIR, TRAIN_PARQUET  # noqa: E402
from copilot.retrieval import bulk_add, get_collection  # noqa: E402


def main() -> None:
    if not TRAIN_PARQUET.exists():
        raise SystemExit(f"Missing {TRAIN_PARQUET}. Run scripts/prepare_data.py first.")

    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Indexing {len(df)} train tickets → {CHROMA_DIR}")

    # Clear existing collection by deleting path contents isn't ideal;
    # upsert is idempotent by ticket id so re-runs are safe.
    ticket_ids = df["ticket_id"].astype(int).tolist()
    texts = df["ticket_text_clean"].astype(str).tolist()
    metadatas = [
        {
            "ticket_id": int(row.ticket_id),
            "type": row.type,
            "queue": row.queue,
            "priority": row.priority,
            "answer": row.answer_clean if hasattr(row, "answer_clean") else row.answer,
            "source": "dataset",
        }
        for row in df.itertuples()
    ]

    count = bulk_add(ticket_ids, texts, metadatas)
    print(f"Done. Collection size: {count}")
    print(f"Verify: {get_collection().count()} docs in Chroma")


if __name__ == "__main__":
    main()
