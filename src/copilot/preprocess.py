"""Text cleaning helpers for tickets and queries."""

from __future__ import annotations

import re

_PLACEHOLDER_RE = re.compile(
    r"<(?:tel_num|email|url|account_num|order_num|ip|date|name|address|"
    r"password|username|phone|amount|id)[^>]*>",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def build_ticket_text(subject: str | None, body: str | None) -> str:
    """Combine subject + body into a single searchable string."""
    subj = (subject or "").strip()
    bod = (body or "").strip()
    if subj and bod:
        return f"{subj}. {bod}"
    return subj or bod


def clean_text(text: str, strict: bool = False) -> str:
    """
    Light clean (default): strip placeholders, collapse whitespace.
    Strict: also lowercase — used for TF-IDF features.
    """
    if not text:
        return ""
    out = _PLACEHOLDER_RE.sub(" ", text)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    if strict:
        out = out.lower()
    return out
