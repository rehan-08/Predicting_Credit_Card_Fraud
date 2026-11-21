# Credit Card Fraud – Kaggle Playground S6E11

The metric is **ROC-AUC**, confirmed on the competition page. The final file is `submission.csv`: 299,984 rows of `id,is_fraud` probabilities. It passes every structural check.

## Result (honest)

| | ROC-AUC |
|---|---:|
| Final stack, nested 5-fold CV (700k rows) | **0.95090** |
| Best single model (LightGBM boosted from a GLM) | 0.95064 |
| Target (public LB) | > 0.95784 |
| Public LB #1 / #2 / #3 (19 Nov 2026) | 0.96003 / 0.95884 / 0.95737 |

**The target has not been achieved in validation.** Our CV is about 0.007 below 0.95784. No Kaggle score exists for this file yet, because it has not been submitted.

Why the gap is probably not closable by honest modelling:
- The public LB uses only about 20% of the test set (≈1,000 frauds). A community study resampled public-sized subsets and found one submission moves about ±0.003 (1 sd) by chance. So 0.95784+ is more than 2 sd above the strongest self-trained CV reported anywhere (~0.9511).
- The top public scores are most likely partly luck on that 20%.
- Pick final submissions by CV, not by public LB.

## Pipeline (`src/`)

| File | Purpose |
|---|---|
| `config.py` | Paths, seed (42), 5 folds |
| `eda.py` | EDA: stats, fraud rate per category, univariate AUCs, `id` check, plots → `reports/eda/` |
| `features.py` | Raw feature frames + hand-made domain features (log amount, ratios, cyclical time, risk interactions…) |
| `te.py` | **Key step**: per-value target encoding of every raw value plus "digit keys" (integer part, int/10, tenths, cents, last 2 decimals, leading digit, digit count). Fitted strictly in-fold. |
| `train_te.py` | Models on encodings: `glm`, `lgb`, `xgb`, `cat`, `lgb_glm`, `xgb_glm` (trees boosted from the GLM logit) |
| `train_nn.py` | RealMLP (pytabkit, GPU) |
| `baselines.py` | LR, HistGB, RF, ExtraTrees, XGBoost, CatBoost, class-weight test, all on raw features |
| `tune_xgb.py` | Optuna (40 trials) for XGB-from-GLM |
| `ensemble.py` | Logistic stacker on logit-ranks of OOF preds, scored by **nested CV**; writes submission |
| `ensemble_combos.py` | LGB+XGB, LGB+CAT, XGB+CAT, three-model blends |
| `leak_check.py`, `validate_submission.py`, `feature_importance.py`, `report.py` | Checks and reports |

All models share `StratifiedKFold(5, shuffle=True, random_state=42)`. Early stopping uses a 10% inner holdout of the training fold, never the scored fold. Test predictions are the average of the 5 fold models. Each fold model is trained on 80% of the data with its own in-fold encoder, so train and test see identical feature construction.

## What mattered (5-fold OOF ROC-AUC)

| Step | AUC |
|---|---:|
| Logistic regression, raw columns | 0.94187 |
| LightGBM 31 leaves, raw | 0.94668 |
| + hand-made domain features | 0.94563 (hurt → dropped) |
| LightGBM 4 leaves (labels ≈ logistic formula, tiny trees win) | 0.94759 |
| GLM on in-fold value + digit-key encodings | 0.94928 |
| 3-leaf LightGBM on raw + encodings | 0.95034 |
| XGBoost depth-2 boosted from GLM logit | 0.95054 |
| … Optuna-tuned (depth 3, colsample 0.87, lr 0.02) | 0.95061 |
| LightGBM 3-leaf boosted from GLM logit | 0.95064 |
| Seed-bagged encodings (3 seeds), XGB-GLM | 0.95074 |
| **Stack of 22 models, nested CV** | **0.95090** |

Things that did **not** help:
- class weighting (LGB 0.94674 vs 0.94759; LR unchanged),
- pair-interaction keys (+0.00003),
- smoothing m=80 vs 20 (±0),
- jointly-fitted one-hot GLM (0.9473 vs 0.9481 on fold 0),
- RealMLP (0.9443),
- RandomForest (0.9328) and ExtraTrees (0.9362).

The full list is in `reports/model_comparison.md` and `reports/experiments.jsonl`.

Final stack at the F1-optimal threshold (0.266), out of fold:
- precision 0.555, recall 0.488, F1 0.519, PR-AUC 0.533,
- confusion matrix [TN, FP, FN, TP] = [683699, 4572, 5989, 5700].

Details per model are in `reports/final_metrics.json`.

## EDA highlights
- No missing values and no duplicate rows (train, or train+test).
- Train and test distributions match.
- `id` carries no signal (AUC 0.4985, flat fraud rate over id deciles), so it is excluded.
- Strongest signals: `merchant_risk_score`, `velocity_score`, `cvv_retry_count`, `auth_method` ("No Authentication" 6.4% fraud vs Biometric 0.5%), `ip_country_mismatch` (16.5%), `is_ai_generated_scam_attempt` (15.2%), `billing_shipping_mismatch`, `is_foreign_transaction`, and merchant category (Gift Cards / Crypto ≈ 7%).
- Generator fingerprints: the **integer part** of `merchant_risk_score` / `velocity_score` carries the signal (GLM coefficient ≈ 1.07, while the full value's encoding gets ≈ 0). Decimal digits and repeated `account_balance_usd` values also carry a small label signal.

## Leakage checks (`reports/leak_check.txt`)
- Label-shuffled pipeline AUC is 0.5012, which is chance level.
- Encoding before the split (deliberately wrong) inflates fold 0 to 0.9797, versus an honest 0.9481. That is why every encoding is nested inside the folds.

## How to run

```powershell
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # only for train_nn.py
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

Put `train.csv`, `test.csv` and `sample_submission.csv` in `data/` first.

> **Windows note:** `lightgbm` must be imported before pandas/sklearn, or it segfaults (OpenMP DLL clash). Every module starts with `import lightgbm`.

`cache/` (≈6 GB of fold encodings) can be deleted at any time; it is rebuilt on demand.

## Next experiments if more score is wanted
1. A target-first language model over serialized rows. Community reports say it is the most diverse strong model (OOF ≈ 0.950, worth about +0.0004 in a stack).
2. RealMLP fed the encoded features, with more epochs and tuned lr. Mine was under-trained (0.944 vs ≈0.950 reported).
3. Pool public OOF banks published on the same folds. Community reports nested CV ≈ 0.9521, but that is other people's work.
4. Submit and compare with CV. Because the public LB noise (±0.003) is large, keep one CV-best file as a final pick.
