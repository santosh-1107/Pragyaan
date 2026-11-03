"""Build retrieval context, call LLM, parse citations."""

from __future__ import annotations

import re

from .llm import generate, is_no_answer
from .retrieval import RetrievedTicket

_CITE_RE = re.compile(r"\[(\d+)\]")


def build_context(hits: list[RetrievedTicket]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[ticket_id={h.ticket_id}] (similarity={h.similarity:.3f})\n"
            f"Q: {h.snippet}\n"
            f"A: {h.answer[:800]}"
        )
    return "\n\n".join(blocks)


def answer_from_hits(question: str, hits: list[RetrievedTicket]) -> tuple[str | None, list[int]]:
    """
    Returns (answer_or_None, cited_ticket_ids).
    None answer means treat as no-match → escalate.
    """
    if not hits:
        return None, []

    context = build_context(hits)
    fallback = hits[0].answer
    raw = generate(question, context, fallback_answer=fallback)

    if is_no_answer(raw):
        return None, []

    cited = [int(x) for x in _CITE_RE.findall(raw)]
    if not cited:
        # If model didn't cite, attach top hit as implicit source
        cited = [hits[0].ticket_id]
    return raw, cited
