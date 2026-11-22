"""GAM-style logistic regression: every feature one-hot (numerics quantile-binned)."""
import lightgbm  # noqa: F401
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from features import load, NUM, CAT, BOOL
from cv import run_cv


def gam_matrix(n_bins=50):
    tr, te = load()
    y = tr["is_fraud"].values.astype(int)
    full = pd.concat([tr.drop(columns="is_fraud"), te], ignore_index=True)
    cols = {}
    for c in NUM:
        if full[c].nunique() <= n_bins:
            cols[c] = full[c].astype(str)
        else:  # bin edges from feature values only (no target)
            cols[c] = pd.qcut(full[c].rank(method="first"), n_bins, labels=False).astype(str)
    for c in CAT + BOOL:
        cols[c] = full[c].astype(str)
    D = pd.DataFrame(cols)
    M = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit_transform(D).tocsr()
    n = len(tr)
    return M[:n], y, M[n:]


class Sp:  # tiny wrapper so cv.run_cv can .iloc a sparse matrix
    def __init__(self, m): self.m = m; self.shape = m.shape
    @property
    def iloc(self): return self
    def __getitem__(self, i): return self.m[i]
    def __len__(self): return self.m.shape[0]


if __name__ == "__main__":
    for nb in [int(a) for a in sys.argv[1:]] or [50]:
        X, y, Xt = gam_matrix(nb)
        for C in [0.05, 0.3]:
            def fp(Xa, ya, Xb, yb, Xt_):
                m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
                m.fit(Xa, ya)
                return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt_)[:, 1]
            run_cv(f"lr_gam_b{nb}_C{C}", fp, Sp(X), y, Xt, verbose=False)
