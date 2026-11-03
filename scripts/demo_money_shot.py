"""Verify the demo money shot: escalate -> resolve -> ask again -> resolved."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.db import ensure_db
from copilot.escalation import resolve_escalation
from copilot.pipeline import ask


def main() -> None:
    ensure_db()
    token = uuid.uuid4().hex[:10]
    q = (
        f"MONEYSHOT_{token}: production zx-ledger control plane unreachable "
        f"and customers cannot checkout on cluster {token}"
    )
    print("1) Ask novel question...")
    r1 = ask(q, log=True)
    print(f"   route={r1.route} escalation_id={r1.escalation_id} "
          f"severity={r1.severity} confidence={r1.confidence}")
    assert r1.route == "escalated", r1

    print("2) Senior resolves...")
    resolution = (
        f"Failover cluster {token} to region B, restart zx-ledger pods, "
        "clear checkout cache. Documented in runbook ZX-77."
    )
    out = resolve_escalation(r1.escalation_id, resolution)
    print(f"   {out}")

    print("3) Ask the same question again...")
    r2 = ask(q, log=True)
    print(f"   route={r2.route} confidence={r2.confidence}")
    print(f"   answer[:200]={ (r2.answer or '')[:200] }")
    assert r2.route == "resolved", r2
    print("MONEY SHOT OK")


if __name__ == "__main__":
    main()
