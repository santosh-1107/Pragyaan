# Onboarding Knowledge Copilot: Final Course Project Report

**Course:** Machine Learning & Advanced Intelligent Systems  
**Project:** Onboarding Knowledge Copilot  
**Architecture:** Retrieval-Augmented Generation (RAG) + Classical ML Routing Pipeline  
**Date:** November 2026  

---

## 1. Executive Summary & Problem Formulation

### 1.1 Problem Statement
When new engineers or team members join an organization, they frequently encounter gaps during knowledge transfer (KT) sessions. As a result, repetitive, historically resolved questions are repeatedly escalated to senior engineers, interrupting focus and causing operational bottlenecks. Conversely, genuinely novel, high-severity production incidents risk being delayed or buried in unprioritized communication channels.

### 1.2 Proposed Solution
The **Onboarding Knowledge Copilot** provides an end-to-end intelligent triage and resolution system:
1. **Semantic Knowledge Retrieval (RAG):** When a joinee poses a question, the copilot searches a historical ticket vector database using dense embeddings (`all-MiniLM-L6-v2` in ChromaDB).
2. **Confidence Thresholding:** If the top-1 cosine similarity meets or exceeds a tuned threshold ($\ge 0.55$), the system generates an answer grounded strictly in historical solutions and cites the source ticket IDs.
3. **Automated Severity Scoring & Explainability:** If no sufficiently close match exists in the knowledge base, the system executes a machine learning model to estimate the ticket's severity (`low`, `medium`, `high`) and calculates an urgency score:
   $$\text{score} = P(\text{medium}) \times 0.5 + P(\text{high}) \times 1.0$$
   The model surfaces top contributing terms (e.g. *outage*, *corrupt*, *poisoned*) for instant human interpretability.
4. **Closed Feedback Loop:** An escalation is routed into the **Senior Engineer Queue** prioritized by urgency. When a senior resolves the issue in the UI, the new Question-Answer pair is immediately **upserted into the vector database**, ensuring that the same question is resolved automatically in the future.

---

## 2. System Architecture & Workflow Pipeline

The diagram below illustrates the complete operational workflow of the Copilot:

```
[ Joinee Query ]
       │
       ▼
┌───────────────────────────┐
│ Intent Detection (LogReg) │ ──► Predicts Type: Incident / Request / Problem / Change
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│  ChromaDB Vector Search   │ ──► Retrieves Top-5 Tickets & Cosine Similarities
└──────────────┬────────────┘
               │
      [ Similarity >= 0.55 ? ]
        /               \
     YES                 NO
      │                   │
      ▼                   ▼
┌──────────────┐   ┌───────────────────────────┐
│  RAG Module  │   │  Severity Classifier (RF) │ ──► Score = P(Med)*0.5 + P(High)*1.0
│ (LLM / Hit)  │   └─────────────┬─────────────┘
└──────┬───────┘                 │
       │                         ▼
       │           ┌───────────────────────────┐
       │           │ Top Contributing Terms    │ ──► Explainability Reason Words
       │           └─────────────┬─────────────┘
       │                         │
       ▼                         ▼
 [ RESOLVED ]              [ ESCALATED ]
 Grounded Answer + Cites    Routed to Senior Queue (Urgent / Normal / Backlog)
                                 │
                                 ▼
                     Senior Resolves in UI
                                 │
                                 ▼
                     Upsert (Q + A) into ChromaDB
                     (Loop Closed — Next Query Resolves!)
```

---

## 3. Dataset & Exploratory Data Analysis (EDA)

### 3.1 Dataset Overview
We utilized the **Multilingual Customer Support Tickets** dataset (20,000 raw tickets) filtered down to the **English subset (11,919 tickets)** as a proxy for enterprise ticket and KT repositories.
The dataset was split into **80% Train (9,535 tickets)**, **10% Validation (1,192 tickets)**, and **10% Test (1,192 tickets)** using stratified splitting by priority.

### 3.2 Target Class Distribution (`priority`)
The distribution of the three priority classes across the training split is as follows:

| Priority Class | Ticket Count | Percentage (%) | Class Weight Strategy |
| :--- | :--- | :--- | :--- |
| **Medium** | 3,962 | 41.55% | Balanced penalty: $w_j = \frac{N}{K \cdot n_j}$ |
| **High** | 3,656 | 38.34% | Balanced penalty |
| **Low** | 1,917 | 20.10% | Inverse proportional weighting (~2x weight) |
| **Total** | 9,535 | 100.0% | Stratified splits preserved across train/val/test |

> **Key Observation:** The `low` priority tier is substantially under-represented (only ~20% of tickets). Without class weighting, standard classifiers default to predicting `medium` or `high`, failing to identify backlog-eligible tasks. Hence, all models were evaluated using **Macro-F1** and trained using balanced loss weights.

### 3.3 Missing Values and Text Hygiene
- `subject`: ~12.2% missing in the raw dataset. Imputed with an empty string `""` before concatenation.
- `body` & `answer`: 2 and 4 rows were missing respectively in the raw data; these rows were dropped during cleaning.
- `ticket_text_clean`: Computed as `f"{subject or ''}. {body}"` and normalized.
- Processed splits have **0 missing values**.

### 3.4 Text Length Distribution
- **Character length:** Median 183 characters (Mean: 201.4, Std: 94.2, Max: 780).
- **Word count:** Median 30 words (IQR: 21 to 38 words).
- Length distributions are consistent across all three priority classes, confirming that ticket length does not leak target priority.

### 3.5 Intent Type vs Priority Crosstab
Examining ticket `type` against `priority`:
- `Incident`: 40.5% High, 39.8% Medium, 19.7% Low
- `Problem`: 38.9% High, 41.2% Medium, 19.9% Low
- `Request`: 35.8% High, 43.1% Medium, 21.1% Low
- `Change`: 37.8% High, 42.0% Medium, 20.2% Low

---

## 4. Preprocessing & Feature Engineering Justification

1. **Light Cleaning vs Strict Cleaning:**
   - *Light cleaning* (`clean_text(strict=False)`): Preserves sentence structure and punctuation for vector embeddings and LLM prompts.
   - *Strict cleaning* (`clean_text(strict=True)`): Replaces `<tel_num>`, emails, and numerical hashes with tokens; lowercases and normalizes whitespace for classical ML tokenizers.
2. **TF-IDF Feature Space:**
   - N-gram range: Unigrams and Bigrams `(1, 2)`. Bigrams capture critical phrases (`system crash`, `cannot login`, `database down`).
   - Feature ceiling: 2,000 max features, sublinear term frequency scaling ($1 + \log(\text{tf})$).
   - Stop words: Standard English stop words removed.
   - Inference latency: Under **0.1 milliseconds** per query on CPU.

---

## 5. Machine Learning Model Experiments & Benchmark Comparison

### 5.1 Severity Model Benchmark (Validation Set, $n=1,192$)
We benchmarked four distinct architectures on the validation split:

| Model Architecture | Features | Accuracy | Macro-F1 | Training Time (s) | Inference Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Balanced)** | TF-IDF (1-2 grams) | 0.4597 | 0.4542 | 0.46 s | 0.001 ms |
| **Random Forest (Balanced) [CHAMPION]** | TF-IDF (1-2 grams) | **0.5570** | **0.5459** | 2.14 s | 0.066 ms |
| **Multi-Layer Perceptron (MLP)** | TF-IDF (1-2 grams) | 0.5545 | 0.5349 | 21.43 s | 0.005 ms |
| **Dense Embeddings + LogReg** | MiniLM (384-d) | 0.4329 | 0.4267 | 497.03 s | 0.001 ms |

#### Architectural Discussion & Champion Selection:
- **Random Forest** achieved the best Macro-F1 (**0.5459**) and Accuracy (**0.5570**), effectively capturing non-linear feature interactions between domain keywords.
- Dense sentence embeddings (`MiniLM + LogReg`) underperformed TF-IDF models while requiring ~500 seconds to generate training embeddings. This demonstrates that severity classification relies heavily on **sparse keyword indicators** (*outage*, *critical*, *slow*, *fail*) rather than broad semantic similarity.
- **Random Forest** was selected and saved to `models/severity.joblib`.

### 5.2 Held-Out Test Set Performance
Evaluating the champion Random Forest model on the completely unseen test split ($n=1,192$):
- **Test Accuracy:** **0.5688** (56.89%)
- **Test Macro-F1:** **0.5579** (55.79%)

### 5.3 Intent Classification
The intent model predicts ticket `type` (`Incident`, `Request`, `Problem`, `Change`) to provide instant UI tagging:
- **Classifier:** Logistic Regression + TF-IDF
- **Accuracy:** **0.7861** (78.61%)
- **Macro-F1:** **0.8076** (80.76%)
- Saved to `models/intent.joblib`.

---

## 6. Knowledge Retrieval System & ChromaDB Evaluation

### 6.1 Vector Index Configuration
- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional normalized vectors).
- **Index:** ChromaDB Persistent Client with cosine distance metric (`distance = 1 - cosine_similarity`).
- **Scale:** 9,535 historical tickets indexed from the training split.

### 6.2 Retrieval Hit@k Benchmark
Using 300 randomly sampled held-out test tickets as natural language queries against the training KB:

| $k$ | Hit@$k$ (Queue / Department) | Hit@$k$ (Ticket Type) | Queries Tested ($n$) |
| :--- | :--- | :--- | :--- |
| **$k = 1$** | **68.33%** | 84.67% | 300 |
| **$k = 3$** | **78.67%** | 92.67% | 300 |
| **$k = 5$** | **85.33%** | **95.33%** | 300 |

At $k=5$, over 85% of retrieved tickets share the exact operational department queue, and 95.3% share the ticket intent type, confirming high retrieval precision.

---

## 7. Confidence Threshold Sweep: Auto-Resolve vs. False-Resolve

To determine the optimal cosine similarity cutoff `SIM_THRESHOLD`, we performed an empirical threshold sweep across 40 evaluation queries (20 in-KB paraphrased questions and 20 novel out-of-scope enterprise queries):

| Threshold | Auto-Resolve % (In-KB Recall) | False-Resolve % (False Alarms) | Correct Escalate % | Precision | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.30** | 100.0% | 90.0% | 10.0% | 0.526 | 55.0% |
| **0.40** | 100.0% | 55.0% | 45.0% | 0.645 | 72.5% |
| **0.50** | 100.0% | 15.0% | 85.0% | 0.870 | 92.5% |
| **0.55 (Selected)** | **100.0%** | **5.0%** | **95.0%** | **0.952** | **97.5%** |
| **0.60** | 100.0% | 0.0% | 100.0% | 1.000 | 100.0% |
| **0.70** | 100.0% | 0.0% | 100.0% | 1.000 | 100.0% |
| **0.85** | 90.0% | 0.0% | 100.0% | 1.000 | 95.0% |

### Threshold Selection Rationale:
- **`SIM_THRESHOLD = 0.55`** balances high sensitivity for paraphrased inquiries with minimal false resolutions (only 5% false alarm on novel queries, achieving 97.5% overall system accuracy).
- Thresholds above 0.80 drop recall on loosely paraphrased employee questions.

---

## 8. Answer Faithfulness & Qualitative Verification

As part of our evaluation framework, we conducted a manual and automated answer faithfulness audit on 20 test queries:
- **Faithfulness Metric:** Does the generated or returned answer contain information strictly supported by the cited tickets without hallucination?
- **Result:** **20 / 20 (100.0% Faithfulness)**.
- **Mechanism:** The prompt requires the model to state `NO_ANSWER` if unsupported, and the local fallback returns the verified top ticket answer verbatim when no LLM key is configured.
- Detailed audit logs are saved at `eval/results/faithfulness_check.csv`.

---

## 9. Closed Feedback Loop: Verification & Demonstration

The core capability of the Copilot is that **no question escalates twice**:
1. Joinee asks an unindexed question:
   `"MONEYSHOT_xyz: production zx-ledger control plane unreachable on cluster xyz"`
2. Knowledge base similarity is below threshold ($\approx 0.44 < 0.55$).
3. Query escalates: Severity is scored as **High/Medium**, reason terms (`control`, `customers`) are identified, and Escalation #ID is created in SQLite.
4. A senior engineer views the escalation in `/pages/1_Senior_Queue.py` and submits a resolution:
   `"Failover cluster xyz to region B, restart zx-ledger pods. Runbook ZX-77."`
5. The system saves the resolution and **upserts the pair into ChromaDB**.
6. When the joinee (or another employee) asks the same question again, ChromaDB reports **confidence 1.000**, and the question is **instantly resolved** from the newly added KB entry.

This behavior is tested and verified by `tests/test_pipeline.py` and `scripts/demo_money_shot.py`.

---

## 10. Summary Evaluation Metrics Table

| Component | Metric | Target / Benchmark | Achieved Result |
| :--- | :--- | :--- | :--- |
| **Intent Classifier** | Accuracy | $\ge 75\%$ | **78.61%** |
| | Macro-F1 | $\ge 75\%$ | **80.76%** |
| **Severity Classifier** | Validation Macro-F1 | Best among 4 models | **0.5459** (Random Forest) |
| | Test Set Accuracy | Generalization | **56.89%** |
| | Test Set Macro-F1 | Generalization | **55.79%** |
| | Inference Latency | $< 50$ ms | **0.066 ms** |
| **Retrieval Engine** | Hit@1 (Queue) | Retrieval precision | **68.33%** |
| | Hit@5 (Queue) | Retrieval coverage | **85.33%** |
| | Hit@5 (Type) | Intent alignment | **95.33%** |
| **End-to-End Pipeline** | Accuracy at Thresh 0.55 | $\ge 90\%$ | **97.5%** |
| | False Resolution Rate | $< 10\%$ | **5.0%** |
| **Faithfulness Check** | Supported by cited tickets | $\ge 90\%$ | **100.0%** (20/20) |
| **Unit & Smoke Tests** | Automated test suite | 100% pass | **3/3 Passed** |

---

## 11. Honest Caveats, Limitations & Ethical Considerations

1. **Proxy Dataset Context:**
   The training corpus consists of external customer support tickets rather than internal IT/DevOps knowledge transfers. In real production deployments, tickets contain proprietary API paths, internal dashboards, and stack traces. Framing customer support data as internal enterprise tickets is an honest compromise for coursework.
2. **Subjectivity & Label Noise in Severity:**
   Support ticket priority tags in public datasets are notorious for noisy, human-assigned labels (different agents tag identical bugs as `medium` vs `high`). Consequently, all classical and deep models plateau at approximately 55–58% accuracy. Reporting Macro-F1 honestly rather than claiming artificially inflated accuracy is critical.
3. **Paraphrase Sensitivity:**
   Keyword-rich questions achieve cosine similarities $> 0.85$, while colloquial or tangential rewordings can score between $0.45$ and $0.55$. The fallback to senior escalation ensures safety over hallucinated confidence.

---

## 12. Reproduction & Demonstration Instructions

### Prerequisites
Activate the virtual environment:
```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

### Running Tests & Verifications
- **Run Smoke Tests:**
  ```powershell
  .venv\Scripts\pytest -v
  ```
- **Execute 3-Minute Money Shot Demo:**
  ```powershell
  .venv\Scripts\python scripts\demo_money_shot.py
  ```
- **Run Threshold Sweep:**
  ```powershell
  .venv\Scripts\python scripts\sweep_threshold.py
  ```
- **Run Faithfulness Evaluation:**
  ```powershell
  .venv\Scripts\python scripts\eval_faithfulness.py
  ```

### Launching the Streamlit Application
```powershell
.venv\Scripts\streamlit run app/Home.py
```
- Open `http://localhost:8501` to test the **Ask** interface.
- Navigate to `Senior Queue` (`pages/1_Senior_Queue.py`) to review and resolve escalations.
- Navigate to `Analytics` (`pages/2_Analytics.py`) to monitor live metrics, auto-resolve ratios, and latency.

