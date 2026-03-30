# Bias and Proxy Discrimination in Automated Resume Screening

**USC DSCI 531 — Fairness in AI | Course Project**
Ameya Bhalgat · Rashi Solanki · Mayuresh Mohnalkar

---

## Overview

This project investigates demographic bias and proxy discrimination in automated resume screening (ARS) systems. Using a semi-synthetic dataset of 2,483 real-world résumés, we inject controlled demographic attributes, train baseline classifiers, and evaluate fairness across multiple metrics. We also compare three categories of bias mitigation strategies.

---

## Dataset

Kaggle Resume Dataset — 2,483 PDFs across 24 job categories.
No real demographic labels or hiring outcomes; both are synthetically generated for controlled experimentation.

---

## Pipeline

```
data/ (PDFs)
  └── Cell 2: Load & preprocess
  └── Cell 3: TF-IDF feature extraction (5,000 features)
  └── Cell 4: Demographic injection + biased label generation
  └── Cell 5: Train/test split (70/30)
  └── Cell 6: Evaluation helpers
  └── Cell 7: Baseline models (LR, RF)
  └── Cell 7b: 5-fold cross-validation
  └── Cell 8: Proxy analysis (remove proxy feature, re-evaluate)
  └── Cell 8b/c: Feature importance
  └── Cell 9: Bias mitigation (pre / in / post-processing)
  └── Cell 9b: ThresholdOptimizer diagnostic
  └── Cell 10/10b: Summary tables
  └── Cell 11/12: Plots
```

---

## Setup

```bash
pip install "scipy==1.15.3" "scikit-learn==1.6.1" "fairlearn==0.13.0"
pip install pandas numpy matplotlib seaborn pdfplumber
```

Run `main.ipynb` cell by cell. Outputs are saved to `figures/`.

---

## Key Findings

- Baseline models show consistent DP gaps (~0.034–0.036), confirming absorbed bias
- Removing the proxy feature reduces disparity by only 23–31% — bias persists through correlated TF-IDF vocabulary
- `proxy_score` ranks 7th in RF importance, below five natural job-domain terms
- In-processing (ExpGrad) achieves the best DP reduction at the cost of F1
- Post-processing (ThresholdOptimizer) inverted disparity direction rather than reducing it

---

## Outputs

| File | Description |
|---|---|
| `figures/summary_table.csv` | Main results table |
| `figures/proxy_comparison.csv` | Proxy ablation results |
| `figures/cv_summary.csv` | Cross-validation results |
| `figures/fairness_tradeoff.png` | F1 vs. DP diff scatter |
| `figures/fairness_metrics_bar.png` | Fairness metrics bar chart |
| `figures/feature_importance.png` | RF feature importance |
