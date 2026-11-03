"""Ask page — joinee asks a question."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copilot.db import ensure_db  # noqa: E402
from copilot.feedback import set_thumbs  # noqa: E402
from copilot.pipeline import ask  # noqa: E402
from copilot.severity import routing_bucket  # noqa: E402

st.set_page_config(page_title="Onboarding Knowledge Copilot", page_icon="🧭", layout="wide")
ensure_db()

st.title("Onboarding Knowledge Copilot")
st.caption("Ask a question grounded in past tickets — or escalate to a senior.")

question = st.text_area("Your question", height=120, placeholder="e.g. How do I reset my VPN credentials?")

col_ask, col_clear = st.columns([1, 4])
with col_ask:
    submit = st.button("Ask", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear"):
        for k in ("last_response", "last_interaction_id", "last_question"):
            st.session_state.pop(k, None)
        st.rerun()

if submit and question.strip():
    with st.spinner("Searching knowledge base…"):
        resp = ask(question.strip())
    st.session_state["last_response"] = resp.model_dump()
    st.session_state["last_question"] = question.strip()
    # interaction id is last inserted — fetch via analytics would be heavy;
    # pipeline logs it; get from DB
    from copilot.db import fetchone

    row = fetchone("SELECT id FROM interactions ORDER BY id DESC LIMIT 1")
    st.session_state["last_interaction_id"] = row["id"] if row else None

data = st.session_state.get("last_response")
if data:
    route = data["route"]
    intent = data.get("intent", "")
    st.markdown(f"**Intent:** `{intent}`  ·  **Route:** `{route}`  ·  **Latency:** {data.get('latency_ms', 0)} ms")

    if route == "resolved":
        st.success("Resolved from knowledge base")
        st.markdown(data.get("answer") or "")
        with st.expander(f"Source tickets (confidence={data.get('confidence', 0):.2f})", expanded=True):
            for s in data.get("sources") or []:
                st.markdown(
                    f"- **Ticket {s['ticket_id']}** — similarity `{s['similarity']:.3f}`  \n"
                    f"  {s['snippet'][:200]}"
                )
    else:
        sev = data.get("severity") or {}
        label = sev.get("label", "?")
        score = sev.get("score", 0)
        terms = sev.get("top_terms") or []
        bucket = routing_bucket(score)
        st.warning(
            f"Escalated to senior queue — ticket **#{data.get('escalation_id')}**  \n"
            f"Severity: **{label}** (score {score:.2f}) → **{bucket}**  \n"
            f"Reason terms: {', '.join(terms) if terms else 'n/a'}"
        )
        if data.get("sources"):
            with st.expander("Nearest related tickets (context for senior)"):
                for s in data["sources"]:
                    st.markdown(
                        f"- **Ticket {s['ticket_id']}** — `{s['similarity']:.3f}` — {s['snippet'][:180]}"
                    )

    iid = st.session_state.get("last_interaction_id")
    if iid:
        c1, c2, _ = st.columns([1, 1, 6])
        with c1:
            if st.button("👍 Helpful"):
                set_thumbs(iid, "up")
                st.toast("Thanks for the feedback!")
        with c2:
            if st.button("👎 Not helpful"):
                set_thumbs(iid, "down")
                st.toast("Feedback recorded.")
