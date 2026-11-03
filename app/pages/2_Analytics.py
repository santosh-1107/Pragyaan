"""Analytics dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from copilot.db import ensure_db  # noqa: E402
from copilot.feedback import analytics_summary  # noqa: E402

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")
ensure_db()

st.title("Analytics")
st.caption("Auto-resolve rate, escalations, latency, and feedback.")

stats = analytics_summary()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total interactions", stats["total"])
c2.metric("Auto-resolve %", f"{stats['auto_resolve_pct']}%")
c3.metric("Avg latency (ms)", stats["avg_latency_ms"])
c4.metric("👍 / 👎 ratio", stats["feedback_ratio"])

st.subheader("Escalations by severity")
sev = stats.get("escalations_by_severity") or {}
if sev:
    st.bar_chart(pd.Series(sev, name="count"))
else:
    st.write("No escalations logged yet.")

st.subheader("Feedback")
st.write(f"Thumbs up: **{stats['thumbs_up']}** · Thumbs down: **{stats['thumbs_down']}**")

st.subheader("Last 20 interactions")
recent = stats.get("recent") or []
if recent:
    st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
else:
    st.write("No interactions yet.")

# Model comparison tables if present
results_dir = ROOT / "eval" / "results"
sev_csv = results_dir / "severity_comparison.csv"
intent_csv = results_dir / "intent_comparison.csv"
if sev_csv.exists():
    st.subheader("Severity model comparison")
    st.dataframe(pd.read_csv(sev_csv), use_container_width=True, hide_index=True)
if intent_csv.exists():
    st.subheader("Intent model")
    st.dataframe(pd.read_csv(intent_csv), use_container_width=True, hide_index=True)
