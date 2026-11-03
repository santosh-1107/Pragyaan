"""Log interactions and thumbs up/down feedback."""

from __future__ import annotations

from .db import ensure_db, execute, fetchall, fetchone
from .schemas import AskResponse


def log_interaction(question: str, response: AskResponse) -> int:
    ensure_db()
    sev_label = response.severity.label if response.severity else None
    sev_score = response.severity.score if response.severity else None
    return execute(
        """
        INSERT INTO interactions
            (question, route, intent, confidence, severity_label, severity_score,
             answer, escalation_id, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            question,
            response.route,
            response.intent,
            response.confidence,
            sev_label,
            sev_score,
            response.answer,
            response.escalation_id,
            response.latency_ms,
        ),
    )


def set_thumbs(interaction_id: int, thumbs: str) -> None:
    if thumbs not in ("up", "down"):
        raise ValueError("thumbs must be 'up' or 'down'")
    ensure_db()
    execute(
        "UPDATE interactions SET thumbs = ? WHERE id = ?",
        (thumbs, interaction_id),
    )
    execute(
        "INSERT INTO feedback (interaction_id, thumbs) VALUES (?, ?)",
        (interaction_id, thumbs),
    )


def analytics_summary() -> dict:
    ensure_db()
    total = fetchone("SELECT COUNT(*) AS n FROM interactions") or {"n": 0}
    n = int(total["n"] or 0)
    resolved = fetchone(
        "SELECT COUNT(*) AS n FROM interactions WHERE route = 'resolved'"
    ) or {"n": 0}
    escalated = fetchone(
        "SELECT COUNT(*) AS n FROM interactions WHERE route = 'escalated'"
    ) or {"n": 0}
    avg_lat = fetchone(
        "SELECT AVG(latency_ms) AS avg_ms FROM interactions"
    ) or {"avg_ms": 0}
    thumbs = fetchall(
        "SELECT thumbs, COUNT(*) AS n FROM interactions WHERE thumbs IS NOT NULL GROUP BY thumbs"
    )
    thumbs_map = {r["thumbs"]: r["n"] for r in thumbs}
    up = int(thumbs_map.get("up", 0))
    down = int(thumbs_map.get("down", 0))
    by_sev = fetchall(
        """
        SELECT severity_label, COUNT(*) AS n
        FROM interactions
        WHERE route = 'escalated' AND severity_label IS NOT NULL
        GROUP BY severity_label
        """
    )
    recent = fetchall(
        """
        SELECT id, question, route, intent, confidence, severity_label,
               latency_ms, thumbs, created_at
        FROM interactions
        ORDER BY id DESC
        LIMIT 20
        """
    )
    return {
        "total": n,
        "auto_resolve_pct": round(100.0 * int(resolved["n"] or 0) / n, 1) if n else 0.0,
        "escalated": int(escalated["n"] or 0),
        "avg_latency_ms": round(float(avg_lat["avg_ms"] or 0), 1),
        "thumbs_up": up,
        "thumbs_down": down,
        "feedback_ratio": round(up / down, 2) if down else (float(up) if up else 0.0),
        "escalations_by_severity": {r["severity_label"]: r["n"] for r in by_sev},
        "recent": recent,
    }
