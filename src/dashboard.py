"""Module 5: three-panel results dashboard.

Implements FR-5.1 .. FR-5.3:
  FR-5.1  One figure with exactly 3 titled panels: ROC curve, attention heatmap,
          latency comparison bar chart.
  FR-5.2  Every panel is rebuilt from the modules' SAVED outputs — nothing is hardcoded.
  FR-5.3  Saved at >= 150 DPI.

Run after risk_model.py, attention.py, rag_index.py and report_generator.py.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from attention import encode_sequence, self_attention
from data_pipeline import ROOT

FIG_DIR = ROOT / 'outputs' / 'figures'
SEQUENCE = 'ACGUACGUACGGCUA'


def build_dashboard(save_path: str | Path = FIG_DIR / 'dashboard.png'):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Panel A inputs: Module 1 saved test predictions -------------------
    artifacts = joblib.load(ROOT / 'outputs' / 'model_artifacts.joblib')
    y_test, y_scores = artifacts['y_test'], artifacts['y_scores']
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # --- Panel B inputs: Module 2 recomputed on the fixed demo sequence ----
    X = encode_sequence(SEQUENCE)
    weights, _ = self_attention(X)

    # --- Panel C inputs: Module 3 retrieval + Module 4 generation latencies
    retrieval_log = json.loads((ROOT / 'outputs' / 'retrieval_latencies.json').read_text())
    mean_rag_latency = float(np.mean(list(retrieval_log.values())))
    meta = json.loads((ROOT / 'outputs' / 'report_meta.json').read_text())
    latencies = {
        'RAG retrieval\n(mean of 5 queries)': mean_rag_latency,
        'Report generation\n(Module 4)': meta['generation_latency_s'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].plot(fpr, tpr, lw=2, color='#0F6E6E', label=f'AUC = {roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_title(f'Panel A — Risk Classifier ROC ({artifacts["model_name"]})')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')

    im = axes[1].imshow(weights, cmap='viridis')
    axes[1].set_xticks(range(len(SEQUENCE)))
    axes[1].set_yticks(range(len(SEQUENCE)))
    axes[1].set_xticklabels(list(SEQUENCE), fontsize=7)
    axes[1].set_yticklabels(list(SEQUENCE), fontsize=7)
    axes[1].set_title('Panel B — mRNA Self-Attention Weights')
    axes[1].set_xlabel('Attending TO position')
    axes[1].set_ylabel('Attending FROM position')
    fig.colorbar(im, ax=axes[1], label='Attention weight')

    bars = axes[2].bar(list(latencies.keys()), list(latencies.values()),
                       color=['#0F6E6E', '#1F3A5F'])
    axes[2].bar_label(bars, fmt='%.3fs')
    axes[2].set_title('Panel C — Pipeline Stage Latency')
    axes[2].set_ylabel('Seconds')

    fig.suptitle('Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform — '
                 'Results Dashboard', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved dashboard -> {save_path} (dpi=150)')


if __name__ == '__main__':
    build_dashboard()
