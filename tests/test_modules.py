"""Tests for the PRD functional requirements.

Run from the repo root after executing the full pipeline:
    python src/risk_model.py
    python src/attention.py
    python src/rag_index.py
    python src/report_generator.py
    pytest tests/ -v
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from attention import (InvalidSequenceError, encode_sequence,  # noqa: E402
                       self_attention)
from data_pipeline import (MissingColumnError, clean, load_and_validate,  # noqa: E402
                           run_pipeline)
from schemas import ClinicalSafetyReport  # noqa: E402

RAW = ROOT / 'data' / 'patient_records.csv'
CLEAN = ROOT / 'data' / 'patient_records_clean.csv'


# --------------------------- Module 1 (FR-1.x) ----------------------------

def test_fr11_missing_column_raises_named_error(tmp_path):
    df = pd.read_csv(RAW).drop(columns=['Biomarker_Level'])
    p = tmp_path / 'bad.csv'
    df.to_csv(p, index=False)
    with pytest.raises(MissingColumnError, match='Biomarker_Level'):
        load_and_validate(p)


def test_fr12_cleaning_leaves_no_missing():
    df = clean(load_and_validate(RAW))
    assert df.isnull().sum().sum() == 0


def test_fr13_standardization():
    run_pipeline(RAW, CLEAN)
    df = pd.read_csv(CLEAN)
    scaled = (df[['Age', 'Biomarker_Level', 'Vaccine_Dose_mcg']]
              - df[['Age', 'Biomarker_Level', 'Vaccine_Dose_mcg']].mean()) \
        / df[['Age', 'Biomarker_Level', 'Vaccine_Dose_mcg']].std()
    assert np.allclose(scaled.mean(), 0, atol=0.1)
    assert np.allclose(scaled.std(), 1, atol=0.1)


def test_fr14_split_reproducible():
    from risk_model import split_data
    df = pd.read_csv(CLEAN)
    a, _, _, _ = split_data(df)
    b, _, _, _ = split_data(df)
    assert a.equals(b)


def test_fr17_model_reloads_with_matching_predictions():
    model = joblib.load(ROOT / 'outputs' / 'risk_model.joblib')
    artifacts = joblib.load(ROOT / 'outputs' / 'model_artifacts.joblib')
    assert np.array_equal(model.predict(artifacts['X_test']),
                          artifacts['model'].predict(artifacts['X_test']))


# --------------------------- Module 2 (FR-2.x) ----------------------------

def test_fr21_encode_shape():
    emb = encode_sequence('ACGU')
    assert emb.shape[0] == 4


def test_fr22_invalid_character_rejected():
    with pytest.raises(InvalidSequenceError):
        encode_sequence('ACGX')


def test_fr23_attention_matrix_shape():
    X = encode_sequence('ACGUACGUACGGCUA')
    weights, output = self_attention(X)
    assert weights.shape == (15, 15)


def test_fr24_softmax_rows_sum_to_one():
    for seq in ('ACGUA', 'UUUUU', 'ACGUACGUACGGCUACGUAC'):
        X = encode_sequence(seq)
        weights, _ = self_attention(X)
        assert np.allclose(weights.sum(axis=1), 1.0)


def test_fr25_heatmap_saved():
    assert (ROOT / 'outputs' / 'figures' / 'attention_heatmap.png').exists()


# --------------------------- Module 3 (FR-3.x) ----------------------------

def test_fr31_chunk_word_lengths():
    from rag_index import load_index
    store = load_index()
    lengths = [len(c.split()) for c in store.get()['documents']]
    assert all(0 < n < 500 for n in lengths)


def test_fr34_query_returns_exactly_k():
    from rag_index import load_index, query_guidelines
    store = load_index()
    chunks, latency = query_guidelines(store, 'storage temperature requirements', k=3)
    assert len(chunks) == 3


def test_fr35_latency_in_range():
    from rag_index import load_index, query_guidelines
    store = load_index()
    _, latency = query_guidelines(store, 'adverse event escalation criteria')
    assert 0 < latency < 5


# --------------------------- Module 4 (FR-4.x) ----------------------------

def test_fr41_schema_rejects_bad_input():
    with pytest.raises(Exception):
        ClinicalSafetyReport(
            Eligible_For_Vaccine=[1, 2],  # wrong type
            Recommended_Dose_mg=0.05,
            Reasoning_Steps=['x'],
            FDA_Safety_Warnings_Cited=['y'],
        )
    with pytest.raises(Exception):
        ClinicalSafetyReport(Recommended_Dose_mg=0.05)  # missing required fields


def test_fr44_dose_guardrail_blocks_overdose():
    from report_generator import enforce_dose_guardrail
    report = ClinicalSafetyReport(
        Eligible_For_Vaccine=True,
        Recommended_Dose_mg=0.15,  # 150 mcg > 100 mcg bound
        Reasoning_Steps=['test'],
        FDA_Safety_Warnings_Cited=['test'],
    )
    guarded = enforce_dose_guardrail(report, candidate_dose_mcg=150.0)
    assert guarded.Eligible_For_Vaccine is False
    assert guarded.Recommended_Dose_mg <= 0.1


def test_fr45_disclaimer_present():
    report = ClinicalSafetyReport(
        Eligible_For_Vaccine=True,
        Recommended_Dose_mg=0.05,
        Reasoning_Steps=['a'],
        FDA_Safety_Warnings_Cited=['b'],
    )
    assert report.Disclaimer and 'not real clinical guidance' in report.Disclaimer


def test_fr46_report_saved_as_json_and_md():
    reports = list((ROOT / 'outputs' / 'reports').glob('*.json'))
    assert reports, 'no reports generated'
    for j in reports:
        assert j.with_suffix('.md').exists()


# --------------------------- Module 5 (FR-5.x) ----------------------------

def test_fr51_dashboard_exists():
    from PIL import Image
    p = ROOT / 'outputs' / 'figures' / 'dashboard.png'
    assert p.exists()
    img = Image.open(p)
    assert img.info.get('dpi', (0, 0))[0] >= 150  # FR-5.3
