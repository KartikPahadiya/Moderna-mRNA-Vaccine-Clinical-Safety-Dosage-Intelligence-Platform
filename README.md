# Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform

An end-to-end applied AI case study that chains four AI layers the way production
systems in regulated industries actually do: a classical ML risk classifier, a
from-scratch Transformer self-attention engine, a retrieval-augmented generation
(RAG) index over regulatory guidelines, and an LLM-powered, schema-validated
clinical safety report generator — finished with a three-panel results dashboard.

> **Scope disclaimer:** this is an individual student engineering project built
> for skill demonstration. Nothing here is, or should be used as, real clinical
> guidance. Every generated report carries this disclaimer automatically.

## Architecture

```
[patient_records.csv]
        |
        v
[Module 1: Data Cleaning + ML Risk Classifier]  --->  risk label + probability
        |
        v
[Module 2: mRNA Sequence Self-Attention Engine] --->  attention heatmap
        |
        v
[Module 3: Regulatory RAG Index (ChromaDB)]     --->  relevant guideline chunks
        |
        v
[Module 4: LLM Structured Report Generator]     --->  ClinicalSafetyReport (JSON + MD)
        |
        v
[Module 5: Three-Panel Dashboard]
```

| Module | Responsibility | Tools |
|---|---|---|
| 1 — Data & ML | Clean patient data, train/evaluate a risk classifier | Pandas, NumPy, Scikit-learn |
| 2 — Attention | Self-attention from scratch over an mRNA sequence | NumPy, Matplotlib |
| 3 — RAG Index | Chunk, embed and semantically search the guideline document | LangChain, ChromaDB, Sentence-Transformers |
| 4 — Report Gen | Combine risk + retrieved facts into a schema-validated JSON report | LangChain, Pydantic, OpenAI API (optional) |
| 5 — Dashboard | ROC curve + attention heatmap + latency chart in one figure | Matplotlib |

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Pick one LLM provider for Module 4 and copy `.env.example` to `.env`:

```bash
copy .env.example .env         # then fill in ONE provider block
```

Supported providers (all through one OpenAI-compatible client — just key +
model in `.env`):

| Provider | Required `.env` lines |
|---|---|
| OpenAI | `PROVIDER=openai` + `OPENAI_API_KEY` (default model `gpt-4o-mini`) |
| Google Gemini | `PROVIDER=gemini` + `GEMINI_API_KEY` (default `gemini-2.5-flash`) |
| OpenRouter | `PROVIDER=openrouter` + `OPENROUTER_API_KEY` (any model id, default `openai/gpt-4o-mini`) |
| NVIDIA NIM | `PROVIDER=nvidia` + `NVIDIA_API_KEY` (default `meta/llama-3.3-70b-instruct`) |

`<PROVIDER>_MODEL=...` optionally overrides the model name. If `PROVIDER` is
omitted, the first provider with an API key set is used automatically. Without
any key, `report_generator.py` falls back to a deterministic, rule-based
generator that produces the same Pydantic-validated report — the pipeline
always runs end-to-end.

## Run the pipeline (in order)

```bash
python src/risk_model.py        # Week 1 — cleans data, trains + saves the classifier
python src/attention.py         # Week 2 — self-attention heatmap + interpretation
python src/rag_index.py         # Week 3 — builds the ChromaDB index, logs retrieval
python src/report_generator.py  # Week 4 — sample patient -> validated JSON + MD report
python src/dashboard.py         # Week 4 — three-panel dashboard.png
```

## Verify

```bash
pytest tests/ -v
```

18 tests cover the PRD's functional requirements FR-1.1 through FR-5.3
(missing-column errors, imputation, standardization, split reproducibility,
softmax row sums, invalid-sequence rejection, retrieval k/latency, schema
validation, the overdose guardrail, the mandatory disclaimer, report file
pairs, dashboard DPI).

## Outputs

- `data/patient_records_clean.csv` — imputed dataset (zero missing values)
- `outputs/risk_model.joblib` — selected classifier (highest test F1)
- `outputs/figures/` — confusion matrices, attention heatmap, `dashboard.png` (150 DPI)
- `outputs/reports/` — sample `ClinicalSafetyReport` as `.json` + `.md`
- `outputs/chroma_store/` — persisted vector index (reloads across restarts)
- `outputs/retrieval_log.md` — test questions, top chunks, latencies

## Notebooks

- `notebooks/01_data_eda.ipynb` — distributions, correlation heatmap, class balance
- `notebooks/02_attention_math.ipynb` — step-by-step Q/K/V walkthrough + multi-head bonus

## Data

`data/patient_records.csv` (500 synthetic records, no real patient-identifiable
information) and `data/mrna_safety_guidelines.txt` (regulatory-style reference)
are provided with the project handout and committed here for reproducibility.

## Safety guardrails (Module 4)

The system prompt enforces four negative constraints; two are also enforced in
code regardless of which generator produced the report:

1. Never recommend a dose contradicting the retrieved guidelines — a hard
   guardrail clamps any dose above the 100 mcg bound and sets
   `Eligible_For_Vaccine = false`.
2. Never invent clinical facts outside the retrieved chunks (prompt constraint;
   the deterministic fallback only quotes retrieved text).
3. Insufficient information -> conservative default + explanation in
   `Reasoning_Steps`.
4. Every report includes the visible student-project disclaimer.
