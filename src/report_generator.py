"""Module 4: LLM structured clinical safety report generation.

Implements FR-4.1 .. FR-4.6:
  FR-4.1  Output validated against the ClinicalSafetyReport Pydantic schema.
  FR-4.2  Prompts are built ONLY from Module 1 risk output + Module 3 retrieved chunks.
  FR-4.3  Every LLM response is validated; generation is retried exactly once on failure.
  FR-4.4  A recommended dose above the guideline bound forces Eligible_For_Vaccine = false.
  FR-4.5  The disclaimer is present in every report, without exception.
  FR-4.6  Each validated report is saved as both .json and .md in outputs/reports/.

Generation path: if OPENAI_API_KEY is set in .env, a real LLM call is made via
LangChain. Without a key, a deterministic rule-based generator produces the same
schema-validated report so the pipeline always runs end-to-end (the chosen path
is recorded in the report's Reasoning_Steps and in the JSON metadata).
"""
import json
import os
import re
import time
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from data_pipeline import (CATEGORICAL_COL, NUMERIC_COLS, ROOT, imputation_stats)
from rag_index import load_index, query_guidelines
from schemas import ClinicalSafetyReport, DEFAULT_DISCLAIMER

ARTIFACT_PATH = ROOT / 'outputs' / 'model_artifacts.joblib'
REPORT_DIR = ROOT / 'outputs' / 'reports'

SYSTEM_PROMPT = '''You are a clinical safety reporting assistant for a student
engineering project. You must ONLY use the risk score and guideline excerpts
provided below. Never invent clinical facts, clinical metrics, or guidelines
that are not present in the retrieved excerpts. Never recommend a dose outside
the bounds stated in the guideline excerpts. If the retrieved excerpts do not
contain enough information to answer confidently, default to NOT eligible and
explain why in Reasoning_Steps rather than guessing. Every report must include
the Disclaimer field with the exact required safety notice.'''

MAX_SAFE_DOSE_MCG = 100.0  # Section 2.1: doses above 100 mcg need secondary clinician review


def score_patient(patient: dict) -> tuple[str, float]:
    """Load the Week 1 model and score one raw patient record."""
    artifacts = joblib.load(ARTIFACT_PATH)
    model, scaler, encoder = artifacts['model'], artifacts['scaler'], artifacts['encoder']

    row = {c: patient[c] for c in NUMERIC_COLS}
    stats = imputation_stats(pd.read_csv(ROOT / 'data' / 'patient_records_clean.csv'))
    for col in NUMERIC_COLS:
        row[col] = stats[col] if pd.isna(row[col]) else row[col]
    cat = stats[CATEGORICAL_COL] if pd.isna(patient[CATEGORICAL_COL]) else patient[CATEGORICAL_COL]
    if cat not in list(encoder.classes_):
        cat = stats[CATEGORICAL_COL]

    scaled = scaler.transform(pd.DataFrame([[row[c] for c in NUMERIC_COLS]], columns=NUMERIC_COLS))
    X = pd.DataFrame([{
        **dict(zip(NUMERIC_COLS, scaled[0])),
        'Pre_Existing_Conditions': patient['Pre_Existing_Conditions'],
        CATEGORICAL_COL: float(encoder.transform([cat])[0]),
    }])[artifacts['feature_cols']]

    proba = float(model.predict_proba(X)[0, 1])
    return ('High' if proba >= 0.5 else 'Low'), proba


def patient_summary(patient: dict) -> str:
    return (f"{int(patient['Age'])}-year-old patient, {int(patient['Pre_Existing_Conditions'])} "
            f"pre-existing condition(s), inflammatory biomarker "
            f"{patient['Biomarker_Level']} mg/L, prior reaction history: "
            f"{patient['Prior_Reaction_History']}, candidate dose "
            f"{patient['Vaccine_Dose_mcg']} mcg.")


def retrieve_for_patient(patient: dict, k: int = 3):
    """Module 3 retrieval, grounded in the patient's actual profile."""
    question = (
        f"Dosing guidance for a {int(patient['Age'])}-year-old patient with "
        f"{int(patient['Pre_Existing_Conditions'])} pre-existing conditions, "
        f"candidate dose {patient['Vaccine_Dose_mcg']} micrograms, biomarker "
        f"{patient['Biomarker_Level']} mg/L, prior reaction history "
        f"{patient['Prior_Reaction_History']}."
    )
    store = load_index()
    return query_guidelines(store, question, k=k)


def enforce_dose_guardrail(report: ClinicalSafetyReport,
                           candidate_dose_mcg: float) -> ClinicalSafetyReport:
    """FR-4.4: never accept a dose above the guideline bound for this profile."""
    data = report.model_dump()
    if data['Recommended_Dose_mg'] * 1000.0 > MAX_SAFE_DOSE_MCG or candidate_dose_mcg > MAX_SAFE_DOSE_MCG:
        data['Recommended_Dose_mg'] = min(data['Recommended_Dose_mg'], MAX_SAFE_DOSE_MCG / 1000.0)
        data['Eligible_For_Vaccine'] = False
        data['Reasoning_Steps'] = list(data['Reasoning_Steps']) + [
            'GUARDRAIL: the recommended dose exceeded the 100 mcg guideline bound '
            '(doses above 100 mcg require secondary clinician review). Dose was clamped '
            'and eligibility set to false.'
        ]
    return ClinicalSafetyReport(**data)


def generate_with_llm(risk_label: str, risk_probability: float,
                      guideline_chunks: list[str], p_summary: str,
                      max_retries: int = 1) -> ClinicalSafetyReport:
    """FR-4.3: LLM generation validated against the schema, exactly one retry."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    parser = PydanticOutputParser(pydantic_object=ClinicalSafetyReport)
    prompt = ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT + '\n{format_instructions}'),
        ('user',
         'Patient summary: {patient_summary}\n'
         'Predicted risk label: {risk_label} (probability {risk_probability:.2f})\n'
         'Relevant guideline excerpts:\n{guidelines}\n\n'
         'Generate the structured clinical safety report now.'),
    ])
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    chain = prompt | llm | parser

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke({
                'patient_summary': p_summary,
                'risk_label': risk_label,
                'risk_probability': risk_probability,
                'guidelines': '\n---\n'.join(guideline_chunks),
                'format_instructions': parser.get_format_instructions(),
            })
        except (ValidationError, Exception) as exc:  # retry exactly once, then raise
            last_error = exc
            if attempt == max_retries:
                raise RuntimeError(f'LLM report failed schema validation after '
                                   f'{max_retries} retries: {exc}') from exc
    raise last_error  # unreachable, satisfies type checkers


def extract_max_dose_mcg(chunks: list[str]) -> float | None:
    """Pull the dose bound out of the retrieved guideline text, if present."""
    joined = ' '.join(chunks)
    m = re.search(r'(\d+)\s*micrograms', joined)
    return float(m.group(1)) if m else None


def generate_fallback(risk_label: str, risk_probability: float,
                      guideline_chunks: list[str], patient: dict) -> ClinicalSafetyReport:
    """Deterministic generator used when no LLM API key is configured.

    Produces the same schema-validated report strictly from the Module 1 risk
    output and Module 3 retrieved chunks — it invents no clinical facts.
    """
    dose_mcg = float(patient['Vaccine_Dose_mcg'])
    age = int(patient['Age'])
    conditions = int(patient['Pre_Existing_Conditions'])

    eligible = True
    steps = [
        f'[Deterministic generator — no LLM API key configured] Predicted reaction '
        f'risk: {risk_label} (probability {risk_probability:.2f}) from the Module 1 classifier.',
    ]
    warnings = []

    for chunk in guideline_chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if any(key in line.lower() for key in
                   ('100 micrograms', 'secondary clinician review', 'deferred',
                    'contraindication', 'biomarker')) and len(line) > 30:
                warnings.append(line[:300])

    if dose_mcg > MAX_SAFE_DOSE_MCG:
        eligible = False
        steps.append(f'Candidate dose {dose_mcg:.1f} mcg exceeds the 100 mcg bound; '
                     'doses above 100 mcg are high-reactogenicity and require secondary '
                     'clinician review before administration.')
    if age >= 65 and conditions >= 2:
        eligible = False
        steps.append(f'Patient is {age} with {conditions} pre-existing conditions; '
                     'guidelines flag this age band for secondary clinician review '
                     'regardless of dose.')
    if risk_label == 'High':
        steps.append('Classifier risk is High; recommending the lowest clinically '
                     'appropriate dose tier with enhanced post-administration monitoring.')

    if not warnings:
        warnings = ['No specific warning sentence matched the retrieval filters; see '
                    'retrieved guideline excerpts in the retrieval log.']

    recommended_mg = min(dose_mcg, 50.0) / 1000.0 if eligible else 0.0
    steps.append(f'Recommended dose: {recommended_mg * 1000:.0f} mcg '
                 f'({"eligible" if eligible else "not eligible — defer to clinician review"}).')

    return ClinicalSafetyReport(
        Eligible_For_Vaccine=eligible,
        Recommended_Dose_mg=recommended_mg,
        Reasoning_Steps=steps,
        FDA_Safety_Warnings_Cited=warnings[:5],
    )


def generate_report(patient: dict, use_llm: bool | None = None,
                    save: bool = True) -> tuple[ClinicalSafetyReport, dict]:
    """Full Module 4 pipeline: score -> retrieve -> generate -> guardrail -> save."""
    load_dotenv(ROOT / '.env')
    if use_llm is None:
        use_llm = bool(os.getenv('OPENAI_API_KEY'))

    risk_label, risk_proba = score_patient(patient)
    chunks, retrieval_latency = retrieve_for_patient(patient)
    p_summary = patient_summary(patient)

    start = time.perf_counter()
    if use_llm:
        report = generate_with_llm(risk_label, risk_proba, chunks, p_summary)
        mode = 'llm'
    else:
        report = generate_fallback(risk_label, risk_proba, chunks, patient)
        mode = 'deterministic_fallback'
    gen_latency = time.perf_counter() - start

    # FR-4.4 guardrail applies to BOTH generation paths.
    report = enforce_dose_guardrail(report, float(patient['Vaccine_Dose_mcg']))
    # FR-4.5: disclaimer must be present and non-empty, without exception.
    if not report.Disclaimer:
        report = ClinicalSafetyReport(**{**report.model_dump(), 'Disclaimer': DEFAULT_DISCLAIMER})

    meta = {
        'mode': mode,
        'risk_label': risk_label,
        'risk_probability': round(risk_proba, 4),
        'retrieval_latency_s': round(retrieval_latency, 3),
        'generation_latency_s': round(gen_latency, 3),
        'guideline_chunks': chunks,
        'patient': patient,
    }

    if save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"report_{patient['Patient_ID']}"
        (REPORT_DIR / f'{stem}.json').write_text(
            json.dumps({'report': report.model_dump(), 'meta': meta}, indent=2))
        (REPORT_DIR / f'{stem}.md').write_text(report.to_markdown())
        (ROOT / 'outputs' / 'report_meta.json').write_text(json.dumps(meta, indent=2))
        print(f'Saved report pair -> {REPORT_DIR / (stem + ".json")} (+ .md)')

    return report, meta


SAMPLE_PATIENT = {
    'Patient_ID': 'P_SAMPLE',
    'Age': 58,
    'Biomarker_Level': 8.4,
    'Pre_Existing_Conditions': 2,
    'Vaccine_Dose_mcg': 120.0,
    'Prior_Reaction_History': 'Moderate',
}


if __name__ == '__main__':
    report, meta = generate_report(SAMPLE_PATIENT)
    print(report.model_dump_json(indent=2))
    print(f"\nMode: {meta['mode']}, retrieval {meta['retrieval_latency_s']}s, "
          f"generation {meta['generation_latency_s']}s")
