import lightgbm  # noqa: F401  must load before sklearn/pandas on Windows (OpenMP DLL clash -> segfault)
"""Shared stratified K-fold runner that saves OOF + test predictions."""
import os, json, time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from config import OOF_DIR, REPORT_DIR, SEED, N_FOLDS

LOG = os.path.join(REPORT_DIR, "experiments.jsonl")


def folds(y, seed=SEED, n=N_FOLDS):
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def run_cv(name, fit_predict, X, y, Xt, seed=SEED, save=True, extra=None, verbose=True):
    """fit_predict(Xtr, ytr, Xva, yva, Xte) -> (pred_va, pred_te)."""
    oof = np.zeros(len(y)); tp = np.zeros(Xt.shape[0]); t0 = time.time(); scores = []
    for k, (a, b) in enumerate(folds(y, seed)):
        pv, pt = fit_predict(X.iloc[a], y[a], X.iloc[b], y[b], Xt)
        oof[b] = pv; tp += pt / N_FOLDS
        scores.append(roc_auc_score(y[b], pv))
        if verbose:
            print(f"  [{name}] fold {k} auc={scores[-1]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    auc = roc_auc_score(y, oof); pr = average_precision_score(y, oof)
    res = dict(name=name, auc=round(auc, 6), fold_std=round(float(np.std(scores)), 6),
               pr_auc=round(pr, 6), n_feat=X.shape[1], secs=round(time.time() - t0), **(extra or {}))
    print("RESULT", json.dumps(res), flush=True)
    if save:
        np.save(os.path.join(OOF_DIR, f"{name}_oof.npy"), oof)
        np.save(os.path.join(OOF_DIR, f"{name}_test.npy"), tp)
        with open(LOG, "a") as f:
            f.write(json.dumps(res) + "\n")
    return oof, tp, auc
