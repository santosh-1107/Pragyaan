"""Evaluate answer faithfulness on 20 test queries."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import EVAL_DIR, EVAL_RESULTS
from copilot.db import ensure_db
from copilot.pipeline import ask
from copilot.retrieval import search


def main() -> None:
    ensure_db()
    EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    queries_path = EVAL_DIR / "test_queries.csv"
    if not queries_path.exists():
        print(f"Missing {queries_path}")
        return

    qdf = pd.read_csv(queries_path)
    # Take first 20 resolved queries (or in-KB queries) to test answer grounding
    in_kb = qdf[qdf["expected_route"] == "resolved"].head(20)

    rows = []
    faithful_count = 0

    print(f"Evaluating faithfulness on {len(in_kb)} queries...\n")
    for idx, row in enumerate(in_kb.itertuples(), start=1):
        q = str(row.question)
        resp = ask(q, log=False)

        route = resp.route
        ans = (resp.answer or "").strip()
        hits = search(q, k=3)
        top_hit = hits[0] if hits else None

        # Check faithfulness:
        # In our pipeline:
        # 1) If Gemini is used, the prompt enforces using ONLY past tickets and citations.
        # 2) If Gemini fallback is used, it returns the top ticket's answer verbatim.
        # An answer is faithful if key informative words/facts in the answer are present in the top retrieved ticket(s).
        if route == "resolved" and ans and top_hit:
            # Check overlap between answer and top hit's text / answer
            ans_lower = ans.lower()
            hit_text = (top_hit.snippet + " " + top_hit.answer).lower()
            
            # If fallback verbatim or substantial word overlap:
            is_verbatim = ans in top_hit.answer or top_hit.answer[:100] in ans
            
            # Check shared content words
            ans_words = set(w for w in ans_lower.split() if len(w) > 3)
            hit_words = set(w for w in hit_text.split() if len(w) > 3)
            overlap = len(ans_words.intersection(hit_words)) / max(len(ans_words), 1)

            is_faithful = is_verbatim or (overlap >= 0.50)
            notes = "Verbatim top hit" if is_verbatim else f"High word overlap ({overlap:.2f})"
        elif route == "escalated":
            # If escalated, it refrained from hallucinating when confidence is borderline
            is_faithful = True
            notes = "Refrained from hallucinating (escalated)"
        else:
            is_faithful = False
            notes = "No answer or ungrounded"

        if is_faithful:
            faithful_count += 1

        rows.append(
            {
                "id": idx,
                "question": q[:80],
                "route": route,
                "confidence": resp.confidence,
                "top_ticket_id": top_hit.ticket_id if top_hit else None,
                "is_faithful": is_faithful,
                "faithfulness_notes": notes,
                "answer_snippet": ans[:100] if ans else "N/A",
            }
        )
        print(f"[{idx}/20] faithful={is_faithful} | route={route} | conf={resp.confidence:.3f} | {notes}")

    df_out = pd.DataFrame(rows)
    faithfulness_pct = round(100 * faithful_count / len(in_kb), 1)
    print(f"\nFaithfulness: {faithful_count}/{len(in_kb)} ({faithfulness_pct}%)")

    out_csv = EVAL_RESULTS / "faithfulness_check.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Detailed check saved to {out_csv}")


if __name__ == "__main__":
    main()

