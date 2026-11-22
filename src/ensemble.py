"""Blend OOF predictions. Scores every blend with nested CV (blend weights fitted on
4/5 of the OOF rows with a different split, scored on the held-out 1/5).

usage: python ensemble.py name1 name2 ... [--out submission.csv]
"""
import lightgbm  # noqa: F401
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import OOF_DIR, ROOT, REPORT_DIR
from features import load


def logit_rank(oof, test):
    """Map OOF to logit of its rank; map test through the OOF empirical CDF."""
    n = len(oof)
    r = rankdata(oof) / (n + 1)
    srt = np.sort(oof)
    rt = (np.searchsorted(srt, test, side="left") + np.searchsorted(srt, test, side="right")) / 2
    rt = np.clip(rt + 0.5, 0.5, n + 0.5) / (n + 1)
    return logit(r), logit(rt)


def load_preds(names):
    O, T = [], []
    for nm in names:
        o = np.load(os.path.join(OOF_DIR, f"{nm}_oof.npy")); t = np.load(os.path.join(OOF_DIR, f"{nm}_test.npy"))
        lo, lt = logit_rank(o, t); O.append(lo); T.append(lt)
    return np.column_stack(O), np.column_stack(T)


def stacker(C=0.01):
    return LogisticRegression(C=C, solver="newton-cholesky", max_iter=200)


def nested_auc(O, y, C=0.01, seed=7):
    p = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(O, y):
        p[b] = stacker(C).fit(O[a], y[a]).decision_function(O[b])
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+"); ap.add_argument("--out", default=None)
    ap.add_argument("--C", type=float, default=0.01)
    args = ap.parse_args()
    tr, te = load(); y = tr["is_fraud"].values.astype(int)
    O, T = load_preds(args.names)
    print("individual OOF AUC:")
    for i, nm in enumerate(args.names):
        print(f"  {nm:45s} {roc_auc_score(y, O[:, i]):.6f}")
    print("rank corr:\n", pd.DataFrame(O, columns=range(len(args.names))).corr("spearman").round(4).to_string())
    print(f"equal-weight mean AUC: {roc_auc_score(y, O.mean(1)):.6f}")
    na = nested_auc(O, y, args.C)
    m = stacker(args.C).fit(O, y)
    w = dict(zip(args.names, np.round(m.coef_[0], 4).tolist()))
    print(f"nested-CV stack AUC: {na:.6f}\nweights: {w}")
    with open(os.path.join(REPORT_DIR, "ensembles.jsonl"), "a") as f:
        f.write(json.dumps(dict(members=args.names, nested_auc=round(na, 6), weights=w)) + "\n")
    if args.out:
        pt = 1 / (1 + np.exp(-m.decision_function(T)))
        sub = pd.DataFrame({"id": te["id"].values, "is_fraud": pt})
        sub.to_csv(os.path.join(ROOT, args.out), index=False)
        print("wrote", args.out, sub.shape)


if __name__ == "__main__":
    main()
