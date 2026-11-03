"""Main ask(question) → AskResponse pipeline."""

from __future__ import annotations

import time

from .config import SIM_THRESHOLD, TOP_K
from .escalation import create_escalation
from .feedback import log_interaction
from .intent import predict_intent
from .rag import answer_from_hits
from .retrieval import search
from .schemas import AskResponse, SourceTicket
from .severity import predict_severity


def ask(question: str, *, log: bool = True) -> AskResponse:
    t0 = time.perf_counter()
    q = (question or "").strip()
    if not q:
        return AskResponse(
            route="escalated",
            intent="Request",
            answer="Please enter a non-empty question.",
            confidence=0.0,
            latency_ms=0,
        )

    # 1) Intent
    try:
        intent = predict_intent(q)
    except FileNotFoundError:
        intent = "Request"

    # 2) Semantic search
    hits = search(q, k=TOP_K)
    top_sim = hits[0].similarity if hits else 0.0
    sources = [
        SourceTicket(ticket_id=h.ticket_id, similarity=round(h.similarity, 4), snippet=h.snippet)
        for h in hits
    ]

    # 3) Confidence check → RAG or escalate
    if hits and top_sim >= SIM_THRESHOLD:
        answer, _cited = answer_from_hits(q, hits)
        if answer is not None:
            resp = AskResponse(
                route="resolved",
                intent=intent,
                answer=answer,
                sources=sources,
                confidence=round(top_sim, 4),
                severity=None,
                escalation_id=None,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            if log:
                log_interaction(q, resp)
            return resp
        # LLM said NO_ANSWER → fall through to escalate

    # 4) Severity + escalation
    try:
        severity = predict_severity(q)
    except FileNotFoundError:
        from .schemas import SeverityInfo

        severity = SeverityInfo(label="medium", score=0.5, top_terms=[])

    neighbors = [
        {
            "ticket_id": h.ticket_id,
            "similarity": round(h.similarity, 4),
            "snippet": h.snippet,
            "answer": h.answer[:400],
        }
        for h in hits
    ]
    esc_id = create_escalation(q, intent, severity, neighbors)

    resp = AskResponse(
        route="escalated",
        intent=intent,
        answer=None,
        sources=sources,
        confidence=round(top_sim, 4),
        severity=severity,
        escalation_id=esc_id,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    if log:
        log_interaction(q, resp)
    return resp
