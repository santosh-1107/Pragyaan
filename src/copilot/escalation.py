"""Create / list / resolve escalations (SQLite) + KB upsert on resolve."""

from __future__ import annotations

import json
from typing import Any

from .db import execute, fetchall, fetchone, ensure_db
from .retrieval import upsert
from .schemas import SeverityInfo


def create_escalation(
    question: str,
    intent: str,
    severity: SeverityInfo,
    neighbors: list[dict[str, Any]],
) -> int:
    ensure_db()
    return execute(
        """
        INSERT INTO escalations
            (question, intent, severity_label, severity_score, reason_terms, neighbors_json, status)
        VALUES (?, ?, ?, ?, ?, ?, 'open')
        """,
        (
            question,
            intent,
            severity.label,
            severity.score,
            json.dumps(severity.top_terms),
            json.dumps(neighbors),
        ),
    )


def list_open_escalations() -> list[dict]:
    ensure_db()
    rows = fetchall(
        """
        SELECT * FROM escalations
        WHERE status = 'open'
        ORDER BY severity_score DESC, created_at ASC
        """
    )
    for r in rows:
        r["reason_terms"] = json.loads(r["reason_terms"] or "[]")
        r["neighbors"] = json.loads(r["neighbors_json"] or "[]")
        r["routing"] = _routing(r["severity_score"] or 0.0)
    return rows


def get_escalation(escalation_id: int) -> dict | None:
    row = fetchone("SELECT * FROM escalations WHERE id = ?", (escalation_id,))
    if not row:
        return None
    row["reason_terms"] = json.loads(row["reason_terms"] or "[]")
    row["neighbors"] = json.loads(row["neighbors_json"] or "[]")
    return row


def resolve_escalation(
    escalation_id: int,
    resolution: str,
    resolved_by: str = "senior",
) -> dict:
    """Mark resolved and upsert Q+A into the vector KB."""
    esc = get_escalation(escalation_id)
    if not esc:
        raise ValueError(f"Escalation {escalation_id} not found")
    if esc["status"] != "open":
        raise ValueError(f"Escalation {escalation_id} is already {esc['status']}")

    execute(
        """
        UPDATE escalations
        SET status = 'resolved',
            resolution = ?,
            resolved_at = datetime('now'),
            resolved_by = ?
        WHERE id = ?
        """,
        (resolution, resolved_by, escalation_id),
    )

    ticket_id = upsert(
        question=esc["question"],
        resolution=resolution,
        ticket_type=esc.get("intent") or "Request",
        priority=esc.get("severity_label") or "medium",
        source="senior",
    )
    return {"escalation_id": escalation_id, "kb_ticket_id": ticket_id, "status": "resolved"}


def _routing(score: float) -> str:
    if score >= 0.70:
        return "Urgent"
    if score >= 0.40:
        return "Normal"
    return "Backlog"
