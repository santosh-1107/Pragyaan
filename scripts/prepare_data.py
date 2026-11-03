"""Filter English tickets, clean, build text, stratified 80/10/10 split."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import (  # noqa: E402
    DATA_PROCESSED,
    DATA_RAW,
    TEST_PARQUET,
    TICKETS_CSV,
    TICKETS_PARQUET,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from copilot.preprocess import build_ticket_text, clean_text  # noqa: E402

HF_DATASET = "Tobi-Bueck/customer-support-tickets"
HF_FILE = "dataset-tickets-multi-lang-4-20k.csv"


def download_if_needed() -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if TICKETS_CSV.exists():
        print(f"Using existing CSV: {TICKETS_CSV}")
        return TICKETS_CSV

    print("CSV not found locally - downloading from Hugging Face...")
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{HF_FILE}"
    try:
        import urllib.request

        urllib.request.urlretrieve(url, TICKETS_CSV)
        print(f"Downloaded to {TICKETS_CSV}")
        return TICKETS_CSV
    except Exception as e:
        print(f"Direct download failed: {e}")
        print("Trying datasets library...")
        try:
            from datasets import load_dataset

            ds = load_dataset(HF_DATASET, split="train")
            df = ds.to_pandas()
            df.to_csv(TICKETS_CSV, index=False)
            print(f"Saved via datasets lib to {TICKETS_CSV}")
            return TICKETS_CSV
        except Exception as e2:
            raise SystemExit(
                f"Could not obtain dataset. Place the CSV at {TICKETS_CSV}\n"
                f"Errors: {e} | {e2}"
            )


def prepare() -> None:
    csv_path = download_if_needed()
    df = pd.read_csv(csv_path)
    print(f"Raw rows: {len(df)}")

    # English only
    df = df[df["language"].astype(str).str.lower() == "en"].copy()
    print(f"English rows: {len(df)}")

    # Fill missing subject; drop missing body/answer
    df["subject"] = df["subject"].fillna("")
    before = len(df)
    df = df.dropna(subset=["body", "answer"])
    df = df[df["body"].astype(str).str.strip() != ""]
    df = df[df["answer"].astype(str).str.strip() != ""]
    print(f"Dropped {before - len(df)} rows with missing body/answer -> {len(df)}")

    # Normalize priority / type
    df["priority"] = df["priority"].astype(str).str.lower().str.strip()
    df["type"] = df["type"].astype(str).str.strip()
    df["queue"] = df["queue"].astype(str).str.strip()

    # Build text fields
    df["ticket_text"] = [
        build_ticket_text(s, b) for s, b in zip(df["subject"], df["body"])
    ]
    df["ticket_text_clean"] = df["ticket_text"].map(lambda t: clean_text(t, strict=False))
    df["ticket_text_strict"] = df["ticket_text"].map(lambda t: clean_text(t, strict=True))
    df["answer_clean"] = df["answer"].astype(str).map(lambda t: clean_text(t, strict=False))

    # Stable ticket ids (0-based within English subset)
    df = df.reset_index(drop=True)
    df["ticket_id"] = df.index.astype(int)

    # Stratified 80/10/10 by priority
    train, temp = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["priority"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp["priority"]
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TICKETS_PARQUET, index=False)
    train.to_parquet(TRAIN_PARQUET, index=False)
    val.to_parquet(VAL_PARQUET, index=False)
    test.to_parquet(TEST_PARQUET, index=False)

    print(f"Saved full English to {TICKETS_PARQUET} ({len(df)})")
    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")
    print("Priority distribution (train):")
    print(train["priority"].value_counts().to_string())
    print("Type distribution (train):")
    print(train["type"].value_counts().to_string())


if __name__ == "__main__":
    prepare()
