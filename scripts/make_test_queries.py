"""Build eval/test_queries.csv: in-KB from train tickets + novel OOS queries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.parquet"
OUT = ROOT / "eval" / "test_queries.csv"

NOVEL = [
    "URGENT: nightly_gl_sync_v3 poisoned the air-gapped HSM vault and SOC2 pipeline is offline",
    "How do we rotate PCI HSM keys for the vault this weekend with zero downtime?",
    "The on-call runbook for multi-region failover is blank in Confluence page GL-404",
    "Who owns undocumented ETL job nightly_gl_sync_v3 and how do we pause it?",
    "Emergency feature flag needed to disable recommender_v9 in checkout-canary",
    "Feature store group churn_v2 has no owners and blocking model deploy",
    "Air-gapped lab DNS is down; how do we rebuild offline synonym dictionary?",
    "Redis sentinel quorum failed in prod-eu; what is the manual recovery procedure?",
    "How do we re-enqueue poison messages from topic billing.dlq.v3?",
    "Checkout-v4 canary stuck at 1% with no dashboard — who owns the cutover checklist?",
    "Who approved temporary firewall hole to partner IP 10.66.12.9 last night?",
    "Custom internal Claude plugin broke after Friday deploy of authz-sidecar",
    "Need process to request SOC2 evidence pack for vendor AcmeLedger",
    "How to provision staging sandbox with synthetic PII without hitting prod DLP?",
    "Kubernetes control plane unreachable in cluster ops-blue — escalate immediately",
    "All CI pipelines failing after undocumented secret rotation of GITHUB_PAT_BOT",
    "Blue-green cutover checklist for billing monolith is empty — production tonight",
    "Leaked personal access token never logged; how do we revoke across all orgs?",
    "Primary payment gateway returning 500s only on PCI subnet — not in KB",
    "Flink job zx_ledger_replay_v2 writing corrupt Avro to lakehouse bronze",
]


def main() -> None:
    train = pd.read_parquet(TRAIN)
    in_kb = train.sample(20, random_state=7)
    rows: list[dict] = []
    for r in in_kb.itertuples():
        q = str(r.ticket_text_clean)
        if len(q) > 180:
            q = q[:180].rsplit(" ", 1)[0]
        rows.append(
            {
                "question": q,
                "expected_route": "resolved",
                "note": f"train_ticket_{r.ticket_id}",
            }
        )
    for q in NOVEL:
        rows.append({"question": q, "expected_route": "escalated", "note": "novel_oos"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} queries to {OUT}")


if __name__ == "__main__":
    main()
