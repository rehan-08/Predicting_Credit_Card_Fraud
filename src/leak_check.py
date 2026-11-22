"""Leakage sanity checks for the in-fold target encoding pipeline.
1. Label shuffle: whole pipeline on permuted labels must score ~0.5.
2. Leaky-vs-proper: encoding fitted on all rows before the split inflates CV (shows why we nest)."""
import lightgbm  # noqa: F401
import numpy as np
from sklearn.metrics import roc_auc_score

from cv import folds
from te import build_all, fold_encode, _enc
from train_te import glm_fit

raw, codes, names, y, n, _ = build_all()
C = codes[:n]
a, b = folds(y)[0]

ys = np.random.default_rng(0).permutation(y)
Ttr, (Tva,) = fold_encode(C[a], ys[a], [C[b]])
print("label-shuffle AUC (expect ~0.5):", round(roc_auc_score(ys[b], glm_fit(Ttr, ys[a], [Tva])[1]), 4))

Ttr, (Tva,) = fold_encode(C[a], y[a], [C[b]])
print("proper in-fold GLM fold0 AUC:", round(roc_auc_score(y[b], glm_fit(Ttr, y[a], [Tva])[1]), 4))

Tall = _enc(C, y.astype(float), C, y.mean(), 20)  # WRONG on purpose: uses validation labels
print("leaky (encode-before-split) fold0 AUC:", round(roc_auc_score(y[b], glm_fit(Tall[a], y[a], [Tall[b]])[1]), 4))
