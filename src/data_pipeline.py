"""Module 1 (part 1): data loading, validation and cleaning.

Implements FR-1.1 and FR-1.2 of the PRD:
  FR-1.1  Validate that all 7 required columns are present (named error otherwise).
  FR-1.2  Impute Biomarker_Level with the median and Prior_Reaction_History with the mode.

Standardization (FR-1.3) is applied inside ``risk_model.py`` so that the
StandardScaler shipped with the model is fit on raw, unscaled units and can
correctly score brand-new patients in Module 4.

Running this file produces ``data/patient_records_clean.csv`` with zero missing values.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = [
    'Patient_ID',
    'Age',
    'Biomarker_Level',
    'Pre_Existing_Conditions',
    'Vaccine_Dose_mcg',
    'Prior_Reaction_History',
    'Reaction_Score',
]
NUMERIC_COLS = ['Age', 'Biomarker_Level', 'Vaccine_Dose_mcg']
CATEGORICAL_COL = 'Prior_Reaction_History'
TARGET_COL = 'Reaction_Score'


class MissingColumnError(ValueError):
    """Raised when the input CSV is missing one or more required columns."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(
            f'Missing required column(s): {", ".join(missing)}. '
            f'Expected columns: {", ".join(REQUIRED_COLUMNS)}'
        )


def load_and_validate(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and validate the schema (FR-1.1)."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnError(missing)
    return df[REQUIRED_COLUMNS]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (FR-1.2).

    Numeric columns (Biomarker_Level ~4% missing) are imputed with the column
    median because the biomarker distribution is right-skewed, so the median is
    a more robust central value than the mean. The categorical column
    Prior_Reaction_History (~3.5% missing) is imputed with the mode since it is
    a low-cardinality categorical variable. Rows are not dropped so that all
    500 patient records remain available for training.
    """
    df = df.copy()

    print('Missing values before cleaning:')
    print(df.isnull().sum().to_string())

    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    if df[CATEGORICAL_COL].isnull().any():
        df[CATEGORICAL_COL] = df[CATEGORICAL_COL].fillna(df[CATEGORICAL_COL].mode()[0])

    assert df.isnull().sum().sum() == 0, 'Cleaning failed: missing values remain'
    return df


def imputation_stats(df: pd.DataFrame) -> dict:
    """Median/mode values used for imputation — reused when scoring new patients."""
    stats = {col: float(df[col].median()) for col in NUMERIC_COLS}
    stats[CATEGORICAL_COL] = str(df[CATEGORICAL_COL].mode()[0])
    return stats


def run_pipeline(raw_path: str | Path = ROOT / 'data' / 'patient_records.csv',
                 clean_path: str | Path = ROOT / 'data' / 'patient_records_clean.csv'
                 ) -> pd.DataFrame:
    """Validate -> impute -> save the cleaned dataset."""
    df = load_and_validate(raw_path)
    df = clean(df)
    Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    print(f'\nSaved cleaned dataset ({df.shape[0]} rows, {df.shape[1]} cols) -> {clean_path}')
    return df


if __name__ == '__main__':
    run_pipeline()
