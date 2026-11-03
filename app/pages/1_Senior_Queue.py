"""Senior engineer escalation queue."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from copilot.db import ensure_db  # noqa: E402
from copilot.escalation import list_open_escalations, resolve_escalation  # noqa: E402

st.set_page_config(page_title="Senior Queue", page_icon="🛠️", layout="wide")
ensure_db()

st.title("Senior Escalation Queue")
st.caption("Open escalations sorted by severity score. Resolve to add the answer to the KB.")

escalations = list_open_escalations()

if not escalations:
    st.info("No open escalations. Nice — the KB is covering questions.")
    st.stop()

# Summary table
rows = [
    {
        "id": e["id"],
        "routing": e.get("routing"),
        "severity": e.get("severity_label"),
        "score": round(float(e.get("severity_score") or 0), 3),
        "intent": e.get("intent"),
        "question": (e.get("question") or "")[:80],
        "created": e.get("created_at"),
    }
    for e in escalations
]
st.dataframe(rows, use_container_width=True, hide_index=True)

ids = [e["id"] for e in escalations]
selected = st.selectbox("Select escalation to resolve", ids, format_func=lambda i: f"#{i}")

esc = next(e for e in escalations if e["id"] == selected)

st.subheader(f"Ticket #{esc['id']} — {esc.get('routing')} / {esc.get('severity_label')}")
st.markdown(f"**Question:** {esc['question']}")
st.markdown(
    f"**Intent:** `{esc.get('intent')}` · **Score:** `{esc.get('severity_score'):.3f}` · "
    f"**Reason terms:** {', '.join(esc.get('reason_terms') or []) or 'n/a'}"
)

st.markdown("#### Nearest related tickets")
for n in esc.get("neighbors") or []:
    st.markdown(
        f"- **#{n.get('ticket_id')}** (`{n.get('similarity', 0):.3f}`) — {n.get('snippet', '')[:160]}\n"
        f"  _{str(n.get('answer', ''))[:200]}_"
    )

resolution = st.text_area("Your resolution (will be added to the knowledge base)", height=150)
if st.button("Resolve & add to KB", type="primary"):
    if not resolution.strip():
        st.error("Please write a resolution before submitting.")
    else:
        result = resolve_escalation(esc["id"], resolution.strip(), resolved_by="senior")
        st.success(
            f"Resolved escalation #{result['escalation_id']} — "
            f"upserted as KB ticket {result['kb_ticket_id']}. "
            "Ask the same question on the Ask page to verify the feedback loop."
        )
        st.rerun()
