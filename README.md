# Onboarding Knowledge Copilot

RAG-powered onboarding assistant: answers joinee questions from a historical ticket archive, or escalates with an ML severity score. Seniors resolve escalations in the UI, and the answer is immediately written back into the vector knowledge base so the same question never escalates twice.

> **Dataset:** Multilingual Customer Support Tickets (English subset, 11,919 tickets) — framed as internal KT history for coursework.

---

## Architecture

```
[ Joinee Query ]
       │
       ▼
[ Intent Classifier (LogReg) ] ──► Predicts Type: Incident / Request / Problem / Change
       │
       ▼
[ ChromaDB Vector Search ] ──► Top-5 past tickets (all-MiniLM-L6-v2)
       │
  Similarity >= 0.55 ?
    ├── YES ──► [ RAG / Grounded Answer ] ──► RESOLVED (Answer + Ticket citations)
    └── NO  ──► [ Severity Model (RandomForest) ] ──► Score + Reason terms ──► ESCALATED
                      │
                      ▼
            [ Senior Queue in UI ] ──► Senior resolves ticket
                      │
                      ▼
            [ Upsert into ChromaDB ] (Closed Loop — Next query resolves!)
```

---

## Quick Start

### 1. Environment Setup
```powershell
# In PowerShell:
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

# Optional: set GEMINI_API_KEY in .env for generative answers
# Without an API key, the system uses the local top-ticket fallback.
```

### 2. Run Tests
```powershell
.venv\Scripts\pytest -v
```
All 3 smoke tests passing:
- `test_resolved_path`: In-KB question resolves with answer and citations
- `test_escalated_path`: Novel urgent question escalates with severity scoring
- `test_feedback_loop_upsert`: Upserting resolved ticket makes the question resolvable

### 3. Run the Demo "Money Shot"
```powershell
.venv\Scripts\python scripts\demo_money_shot.py
```
Demonstrates asking a novel urgent question $\rightarrow$ escalated $\rightarrow$ senior resolves $\rightarrow$ asked again $\rightarrow$ resolved from KB.

### 4. Run the Streamlit Application
```powershell
.venv\Scripts\streamlit run app/Home.py
```
- **Home (`app/Home.py`):** Joinee asks questions, views citations, and submits 👍 / 👎 feedback.
- **Senior Queue (`app/pages/1_Senior_Queue.py`):** Open escalations sorted by severity score with nearest tickets context and "Resolve & add to KB" action.
- **Analytics (`app/pages/2_Analytics.py`):** Live dashboard showing auto-resolve rate, latency, severity breakdown, and model benchmarks.

---

## Evaluations & Deliverables

- **Final Course Report:** See [`REPORT.md`](file:///c:/5th%20sem/ML_prj/REPORT.md) for full methodology, EDA, benchmark tables, and discussion.
- **Severity Experiments Notebook:** [`notebooks/severity_experiments.ipynb`](file:///c:/5th%20sem/ML_prj/notebooks/severity_experiments.ipynb) (24 cells with EDA, distributions, heatmaps, model comparison bar plots, confusion matrices, and explainability).
- **Threshold Sweep:** Run `python scripts/sweep_threshold.py` (outputs to `eval/results/threshold_sweep.png` and `threshold_sweep.csv`).
- **Answer Faithfulness Audit:** Run `python scripts/eval_faithfulness.py` (20/20 answers 100% grounded in `eval/results/faithfulness_check.csv`).
- **Model Files:** Saved in `models/` (`severity.joblib`, `intent.joblib`, `tfidf.joblib`).
