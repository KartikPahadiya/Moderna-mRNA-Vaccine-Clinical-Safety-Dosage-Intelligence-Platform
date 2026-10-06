"""Module 1 (part 2): risk classifier training and evaluation.

Implements FR-1.4 .. FR-1.7:
  FR-1.4  80/20 stratified split on Reaction_Score with fixed seed.
  FR-1.5  Train Logistic Regression and Random Forest on identical data.
  FR-1.6  Classification report + confusion matrix for both models, saved to outputs/.
  FR-1.7  Select the higher-F1 model, serialize to outputs/risk_model.joblib.

Also saves ROC-curve inputs and the fitted scaler/encoder so Module 4 and the
dashboard can reproduce predictions without retraining.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_pipeline import (CATEGORICAL_COL, NUMERIC_COLS, ROOT, TARGET_COL,
                           run_pipeline)

FIG_DIR = ROOT / 'outputs' / 'figures'
MODEL_PATH = ROOT / 'outputs' / 'risk_model.joblib'
ARTIFACT_PATH = ROOT / 'outputs' / 'model_artifacts.joblib'
SEED = 42
FEATURE_COLS = NUMERIC_COLS + ['Pre_Existing_Conditions', CATEGORICAL_COL]


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    encoder = LabelEncoder()
    df[CATEGORICAL_COL] = encoder.fit_transform(df[CATEGORICAL_COL])
    return df, encoder


def split_data(df: pd.DataFrame):
    """FR-1.4: 80/20 stratified split with a fixed seed."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def train_models(X_train, y_train):
    """FR-1.5: both models on identical training data."""
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate(models, X_test, y_test):
    """FR-1.6: classification report + confusion matrix for each model."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        results[name] = f1

        print(f'\n--- {name} (test F1 = {f1:.3f}) ---')
        print(classification_report(y_test, preds, target_names=['Low risk', 'High risk']))

        # Save text artifacts
        (ROOT / 'outputs' / f'{name}_classification_report.txt').write_text(
            classification_report(y_test, preds, target_names=['Low risk', 'High risk'])
        )

        # Confusion matrix figure
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Low risk', 'High risk'],
                    yticklabels=['Low risk', 'High risk'], ax=ax)
        ax.set_title(f'Confusion Matrix — {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        fig.tight_layout()
        fig.savefig(FIG_DIR / f'confusion_matrix_{name}.png', dpi=150)
        plt.close(fig)
    return results


def save_best_model(models, results, X_test, y_test, scaler, encoder):
    """FR-1.7: select by test F1 and serialize."""
    best_name = max(results, key=results.get)
    best_model = models[best_name]

    joblib.dump(best_model, MODEL_PATH)

    # Reload check: predictions must match the original run.
    reloaded = joblib.load(MODEL_PATH)
    assert np.array_equal(reloaded.predict(X_test), best_model.predict(X_test)), \
        'Reloaded model predictions differ from original run'

    # Persist everything the dashboard / Module 4 need.
    y_scores = best_model.predict_proba(X_test)[:, 1]
    joblib.dump({
        'model_name': best_name,
        'model': best_model,
        'scaler': scaler,
        'encoder': encoder,
        'X_test': X_test,
        'y_test': y_test,
        'y_scores': y_scores,
        'f1_scores': results,
        'feature_cols': FEATURE_COLS,
    }, ARTIFACT_PATH)

    comparison = (
        f'# Model comparison (test-set F1)\n\n'
        f'| Model | F1 score |\n|---|---|\n'
        + ''.join(f'| {k} | {v:.3f} |\n' for k, v in results.items())
        + f'\n**Selected model: `{best_name}`** (highest test F1).\n'
    )
    (ROOT / 'outputs' / 'model_comparison.md').write_text(comparison)
    print(f'\nSelected model: {best_name} (F1 = {results[best_name]:.3f})')
    print(f'Saved -> {MODEL_PATH}')
    return best_name, best_model, y_scores


def main():
    df = run_pipeline()  # produces data/patient_records_clean.csv (imputed, raw units)
    df, encoder = encode_categorical(df)

    # FR-1.3: standardize numeric features to zero mean / unit variance before
    # training. The scaler is fit on raw (unscaled) units here so Module 4 can
    # score brand-new raw patient records with it.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(df[NUMERIC_COLS])
    df = df.copy()
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    check = df[NUMERIC_COLS].agg(['mean', 'std']).T
    print('\nPost-scaling check (mean ~ 0, std ~ 1):')
    print(check.round(4).to_string())

    X_train, X_test, y_train, y_test = split_data(df)
    # Determinism check (FR-1.4): re-running the split yields identical rows.
    X_train2, _, _, _ = split_data(df)
    assert X_train.equals(X_train2), 'Split is not reproducible with the same seed'

    models = train_models(X_train, y_train)
    results = evaluate(models, X_test, y_test)
    save_best_model(models, results, X_test, y_test, scaler, encoder)


if __name__ == '__main__':
    main()
