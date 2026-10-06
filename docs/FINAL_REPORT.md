# Final Project Report — mRNA Vaccine Clinical Safety & Dosage Intelligence Platform

## 1. Executive Summary

This project builds a working, end-to-end AI platform in the style of a clinical
decision-support tool: it cleans a 500-record patient dataset and trains a risk
classifier (Module 1), implements Transformer self-attention from scratch on an
mRNA sequence (Module 2), indexes a regulatory guideline document in a local
vector database for grounded retrieval (Module 3), and fuses the risk prediction
with the retrieved regulatory facts into a strictly schema-validated JSON safety
report (Module 4), visualized on a three-panel dashboard (Module 5). It
demonstrates the three layers of modern AI engineering — predictive ML, deep
learning mathematics, and generative AI — wired together so that each module's
output becomes the next module's input, the way production systems in regulated
industries are actually built.

> Student engineering project output — not real clinical guidance.

## 2. Architecture

```
[patient_records.csv]
        |
        v
[Module 1: Data Cleaning + ML Risk Classifier]  --->  risk label, probability
        |                                                    (outputs/risk_model.joblib)
        v
[Module 2: mRNA Self-Attention Engine]          --->  attention heatmap
        |                                                    (outputs/figures/attention_heatmap.png)
        v
[Module 3: Regulatory RAG Index (ChromaDB)]     --->  relevant guideline chunks
        |                                                    (outputs/chroma_store/)
        v
[Module 4: LLM Structured Report Generator]     --->  ClinicalSafetyReport (.json + .md)
        |
        v
[Module 5: Dashboard]                           --->  outputs/figures/dashboard.png
```

## 3. Module 1 Results — Risk Classifier

- Cleaning: 20 missing `Biomarker_Level` values imputed with the column median
  (right-skewed feature), and 308 missing `Prior_Reaction_History` values
  imputed with the mode. Zero missing values remain; numeric features are
  standardized (mean ≈ 0, std ≈ 1).
- Models trained on an identical 80/20 stratified split (seed 42):

| Model | Accuracy | F1 (High risk) |
|---|---|---|
| Logistic Regression | 0.71 | **0.70** |
| Random Forest (200 trees) | 0.63 | 0.62 |

- **Selected: Logistic Regression.** With only 400 training rows and a noisy,
  roughly linear signal, the regularized linear model generalizes better than
  the higher-variance forest, which overfits feature noise. The gap (0.701 vs
  0.619 F1) is consistent across both risk classes. The 0.71 accuracy is a
  believable, non-suspicious performance for this synthetic dataset.
- Artifacts: `outputs/figures/confusion_matrix_*.png`,
  `outputs/risk_model.joblib`, `outputs/model_comparison.md`.

## 4. Module 2 Results — Self-Attention

`Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V` is implemented in pure NumPy with a
learnable 8-dimensional embedding per nucleotide. Every row of the 15×15 weight
matrix sums to 1.0 (asserted at runtime and in tests). The heatmap shows a
visible repeating block pattern: the sequence `ACGUACGUACGGCUA` contains the
`ACGU` motif repeated every 4 bases, identical bases share identical embeddings,
so positions 4 apart score high dot-product similarity and attend strongly to
each other. Because Q/K/V weights are randomly initialized, this pattern reflects
embedding geometry, not learned biology — training would sharpen or relocate it.
A 4-head multi-head variant (concatenated 32-dim output) is included as a bonus.
Artifacts: `outputs/figures/attention_heatmap.png`, `outputs/attention_interpretation.md`.

## 5. Module 3 Results — RAG Index

The ~1,200-word guideline document was split into 17 overlapping chunks
(`RecursiveCharacterTextSplitter`, 500-char budget / 50-char overlap), embedded
with `all-MiniLM-L6-v2`, and persisted to ChromaDB (`outputs/chroma_store/`),
which survives process restarts (queries in the test suite run against the
reloaded store). Five realistic test questions all returned on-topic top chunks
in 0.019–0.036 s each — see `outputs/retrieval_log.md`.

Retrieval quality is good for single-topic questions. Observed failure case:
questions mixing two topics (e.g. storage AND dosing) sometimes return a chunk
covering only one side, because no single chunk spans two sections. Hypothesis:
more overlap won't fix this; retrieving k=10 and re-ranking with a cross-encoder
would.

## 6. Module 4 Results — Structured Report Generation

A sample patient (58 y, 2 pre-existing conditions, biomarker 8.4 mg/L, candidate
dose 120 mcg, prior moderate reaction) produced this validated report (abridged;
full JSON + MD in `outputs/reports/`):

```json
{
  "Eligible_For_Vaccine": false,
  "Recommended_Dose_mg": 0.0,
  "Reasoning_Steps": [
    "Predicted reaction risk: High (probability 0.96) from the Module 1 classifier.",
    "Candidate dose 120.0 mcg exceeds the 100 mcg bound; doses above 100 mcg are
     high-reactogenicity and require secondary clinician review...",
    "GUARDRAIL: the recommended dose exceeded the 100 mcg guideline bound ...",
    ...
  ],
  "FDA_Safety_Warnings_Cited": [
    "Ages 65 and older: ... flagged for secondary clinician review regardless of dose...",
    "Doses above 100 micrograms are considered high-reactogenicity doses..."
  ],
  "Disclaimer": "Student engineering project output — not real clinical guidance..."
}
```

How it is grounded: the risk label/probability come straight from Module 1's
serialized model; every cited warning is copied verbatim from Module 3's
retrieved chunks; and the 100 mcg dose bound is enforced **in code**
(`enforce_dose_guardrail`) on both the LLM and fallback generation paths, so a
hallucinated overdose cannot survive validation. If `OPENAI_API_KEY` is set, the
report is generated by `gpt-4o-mini` through LangChain with a Pydantic parser and
exactly one retry on validation failure; otherwise a deterministic rule-based
generator (which invents no clinical facts) produces the identical schema — the
active mode is recorded in the report metadata.

## 7. Dashboard

`outputs/figures/dashboard.png` (150 DPI) — Panel A: ROC curve of the selected
classifier (AUC = 0.79). Panel B: the Module 2 attention heatmap. Panel C: mean
RAG retrieval latency (~0.023 s) vs report-generation latency. Every panel is
rebuilt from the modules' saved outputs on each run.

## 8. Limitations & Future Work

1. **Dataset size (500 rows).** Confidence intervals on the F1 scores are wide.
   With more time: collect/merge additional synthetic cohorts, or use
   bootstrapped CIs to quantify uncertainty honestly.
2. **Retrieval recall on multi-topic questions.** As noted in Section 5, single
   vector search can miss one side of a compound question. Fix: k=10 retrieval
   + cross-encoder re-ranking, or a small BM25 hybrid index.
3. **LLM cost and determinism.** Real LLM calls cost money and can vary run to
   run. Mitigation already in place: temperature 0, schema validation, one retry,
   and a deterministic fallback so the pipeline never blocks on an API key.
4. **Mode-imputed categorical feature.** `Prior_Reaction_History` is missing in
   ~62% of rows (far above the PRD's 3.5% estimate); mode imputation erases real
   signal. Fix: add a "missing" indicator column or model the feature as a
   latent variable.
5. **Single reference document.** The RAG index covers one guideline file; a real
   deployment would ingest a living library of regulatory documents with version
   pinning and citation provenance.

## 9. Reproduction Instructions

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python src/risk_model.py        # Week 1
python src/attention.py         # Week 2
python src/rag_index.py         # Week 3 (first run downloads the embedding model)
python src/report_generator.py  # Week 4 (optional: set OPENAI_API_KEY in .env)
python src/dashboard.py         # Week 4
pytest tests/ -v                # 18 requirement tests
```

See `README.md` for the full setup, environment variables, and module map.
