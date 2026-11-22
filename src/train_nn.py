"""RealMLP (pytabkit) on raw features + digit keys as categoricals, same 5 folds.
Model selection / early stopping uses a 10% inner holdout of the training fold.

usage: python train_nn.py [--seed S] [--epochs E]
"""
import lightgbm  # noqa: F401
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import OOF_DIR, SEED
from cv import folds, LOG
from features import load, NUM, CAT, BOOL
from te import CONT, TWO_DEC


def frame_allcat(min_count=5):
    """Every raw value and digit key as a categorical (rare values merged)."""
    from te import key_frame
    tr, te = load()
    y = tr["is_fraud"].values.astype(int)
    full = pd.concat([tr.drop(columns="is_fraud"), te], ignore_index=True)
    codes, names = key_frame(full)
    X = full[NUM].astype(float).copy()
    cats = []
    for j, nm in enumerate(names):
        c = codes[:, j]
        cnt = np.bincount(c[: len(tr)], minlength=c.max() + 1)
        c = np.where(cnt[c] >= min_count, c, -1)
        X["k_" + nm] = pd.Categorical(c); cats.append("k_" + nm)
    n = len(tr)
    return X.iloc[:n].reset_index(drop=True), y, X.iloc[n:].reset_index(drop=True), cats


def frame():
    tr, te = load()
    y = tr["is_fraud"].values.astype(int)
    full = pd.concat([tr.drop(columns="is_fraud"), te], ignore_index=True)
    X = full[NUM].astype(float).copy()
    for c in BOOL:
        X[c] = full[c].astype(int).astype(float)
    cats = list(CAT)
    for c in CAT:
        X[c] = full[c].astype("category")
    for c in CONT:
        v = np.round(full[c].values * 100).astype(np.int64)
        X[f"{c}__tenth"] = pd.Categorical((v // 10) % 10); cats.append(f"{c}__tenth")
        if c in TWO_DEC:
            X[f"{c}__cent"] = pd.Categorical(v % 10); cats.append(f"{c}__cent")
    # low-cardinality numerics also as categories (lets the net learn per-value effects)
    for c in ["merchant_risk_score", "velocity_score", "customer_age", "card_age_months", "txn_count_last_24h"]:
        X[f"{c}__cat"] = pd.Categorical(full[c].astype(str)); cats.append(f"{c}__cat")
    n = len(tr)
    return X.iloc[:n].reset_index(drop=True), y, X.iloc[n:].reset_index(drop=True), cats


def main():
    from pytabkit import RealMLP_TD_Classifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--allcat", action="store_true")
    args = ap.parse_args()
    X, y, Xt, cats = frame_allcat() if args.allcat else frame()
    oof = np.zeros(len(y)); tp = np.zeros(len(Xt)); t0 = time.time()
    for k, (a, b) in enumerate(folds(y)):
        ia, ib = train_test_split(a, test_size=0.1, stratify=y[a], random_state=args.seed)
        m = RealMLP_TD_Classifier(device="cuda", random_state=args.seed + k, n_epochs=args.epochs,
                                  batch_size=1024, val_metric_name="1-auc_ovr", use_early_stopping=False,
                                  verbosity=0, n_cv=1)
        m.fit(X.iloc[ia], y[ia], X.iloc[ib], y[ib], cat_col_names=cats)
        pv = m.predict_proba(X.iloc[b])[:, 1]; pt = m.predict_proba(Xt)[:, 1]
        oof[b] = pv; tp += pt / 5
        print(f"  fold {k} auc={roc_auc_score(y[b], pv):.5f} ({time.time()-t0:.0f}s)", flush=True)
    name = f"realmlp{'_allcat' if args.allcat else ''}_s{args.seed}_e{args.epochs}"
    auc = roc_auc_score(y, oof)
    res = dict(name=name, auc=round(auc, 6), secs=round(time.time() - t0))
    print("RESULT", json.dumps(res), flush=True)
    np.save(os.path.join(OOF_DIR, f"{name}_oof.npy"), oof); np.save(os.path.join(OOF_DIR, f"{name}_test.npy"), tp)
    with open(LOG, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
