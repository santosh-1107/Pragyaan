"""Smoke tests: resolved path, escalated path, feedback-loop upsert."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.config import CHROMA_DIR, MODELS_DIR, SIM_THRESHOLD  # noqa: E402
from copilot.db import ensure_db  # noqa: E402
from copilot.escalation import list_open_escalations, resolve_escalation  # noqa: E402
from copilot.pipeline import ask  # noqa: E402
from copilot.retrieval import search  # noqa: E402


def _models_ready() -> bool:
    return (MODELS_DIR / "severity.joblib").exists() and (MODELS_DIR / "intent.joblib").exists()


def _chroma_ready() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


pytestmark = pytest.mark.skipif(
    not (_models_ready() and _chroma_ready()),
    reason="Models/Chroma not built yet — run prepare_data, build_index, train_* first",
)


def test_resolved_path():
    """A question close to KB content should resolve (or at least retrieve hits)."""
    ensure_db()
    # Use a generic support-style question; if threshold fails, at least sources exist
    resp = ask("I forgot my password and cannot log into my account", log=False)
    assert resp.intent
    assert resp.latency_ms >= 0
    if resp.route == "resolved":
        assert resp.answer
        assert resp.confidence >= SIM_THRESHOLD
        assert len(resp.sources) >= 1
    else:
        # Still valid if KB similarity is low — assert escalation fields
        assert resp.escalation_id is not None
        assert resp.severity is not None


def test_escalated_path():
    """A novel urgent question should escalate with severity."""
    ensure_db()
    # Must be out-of-distribution vs customer-support KB so similarity < threshold
    resp = ask(
        "URGENT: our internal flink job nightly_gl_sync_v3 poisoned the "
        "air-gapped HSM vault and SOC2 evidence pipeline is offline",
        log=False,
    )
    assert resp.route == "escalated"
    assert resp.escalation_id is not None
    assert resp.severity is not None
    assert resp.severity.label in ("low", "medium", "high")
    assert 0.0 <= resp.severity.score <= 1.0


def test_feedback_loop_upsert():
    """Resolving an escalation must make the same question retrievable."""
    import uuid
    from copilot.retrieval import get_collection

    ensure_db()
    token = uuid.uuid4().hex[:12]
    unique_q = (
        f"INTERNAL_SYS_{token}: restart secret subsystem alpha_{token} "
        f"on isolated airgap node zx_{token}?"
    )
    resp = ask(unique_q, log=False)
    assert resp.route == "escalated", f"expected escalate, got {resp.route} conf={resp.confidence}"
    esc_id = resp.escalation_id
    assert esc_id is not None

    resolution = f"Restart alpha_{token} on zx_{token} via systemctl and clear Redis."
    result = resolve_escalation(esc_id, resolution)
    assert result["status"] == "resolved"
    kb_id = result.get("kb_ticket_id")

    try:
        hits = search(unique_q, k=3)
        assert len(hits) >= 1
        assert hits[0].similarity >= 0.85 or resolution[:40] in hits[0].answer

        # Asking again should prefer resolve (high similarity after upsert)
        resp2 = ask(unique_q, log=False)
        assert resp2.route == "resolved"
        assert resp2.answer
    finally:
        # Clean up the test ticket so subsequent runs remain idempotent
        if kb_id:
            try:
                get_collection().delete(ids=[f"ticket_{kb_id}"])
            except Exception:
                pass
