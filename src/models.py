import lightgbm  # noqa: F401  must load before sklearn/pandas on Windows (OpenMP DLL clash -> segfault)
"""Model factories returning fit_predict callables for cv.run_cv."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from config import SEED

LGB_DEFAULT = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                   reg_alpha=0.0, max_bin=255, n_estimators=5000, verbose=-1, n_jobs=20)
XGB_DEFAULT = dict(learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
                   colsample_bytree=0.5, reg_lambda=1.0, reg_alpha=0.0, n_estimators=5000,
                   tree_method="hist", device="cuda", max_cat_to_onehot=1, enable_categorical=True,
                   eval_metric="auc", early_stopping_rounds=200, max_bin=256)
CB_DEFAULT = dict(learning_rate=0.05, depth=6, l2_leaf_reg=3, iterations=5000, eval_metric="AUC",
                  od_type="Iter", od_wait=200, task_type="GPU", border_count=254, verbose=0)


def lgbm(params=None, seed=SEED, es=200):
    p = {**LGB_DEFAULT, **(params or {}), "seed": seed, "metric": "auc"}
    n_est = p.pop("n_estimators")
    def fp(Xa, ya, Xb, yb, Xt):
        # lgb.train used directly: sklearn eval_set path segfaults in lightgbm 4.7 on Windows
        da = lgb.Dataset(Xa, ya.astype(np.float64))
        db = lgb.Dataset(Xb, yb.astype(np.float64), reference=da)
        m = lgb.train(p, da, n_est, valid_sets=[db],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
        fp.best_iters.append(m.best_iteration); fp.models.append(m)
        return m.predict(Xb, num_iteration=m.best_iteration), m.predict(Xt, num_iteration=m.best_iteration)
    fp.best_iters, fp.models = [], []
    return fp


def xgbm(params=None, seed=SEED):
    p = {**XGB_DEFAULT, **(params or {}), "random_state": seed}
    def fp(Xa, ya, Xb, yb, Xt):
        m = xgb.XGBClassifier(**p)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        fp.best_iters.append(m.best_iteration); fp.models.append(m)
        return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1]
    fp.best_iters, fp.models = [], []
    return fp


def catb(params=None, seed=SEED):
    p = {**CB_DEFAULT, **(params or {}), "random_seed": seed}
    def fp(Xa, ya, Xb, yb, Xt):
        cats = [c for c in Xa.columns if str(Xa[c].dtype) == "category"]
        conv = lambda d: d.assign(**{c: d[c].astype(str) for c in cats})
        Xa, Xb, Xt = conv(Xa), conv(Xb), conv(Xt)
        m = CatBoostClassifier(**p)
        m.fit(Pool(Xa, ya, cat_features=cats), eval_set=Pool(Xb, yb, cat_features=cats),
              use_best_model=True)
        fp.best_iters.append(m.get_best_iteration()); fp.models.append(m)
        return m.predict_proba(Pool(Xb, cat_features=cats))[:, 1], \
            m.predict_proba(Pool(Xt, cat_features=cats))[:, 1]
    fp.best_iters, fp.models = [], []
    return fp
