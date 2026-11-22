"""Reference models on raw features (feature set 'base') for the comparison table.
usage: python baselines.py lr hgb rf et xgb cat lgb_balanced"""
import lightgbm  # noqa: F401
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from features import build, CAT
from cv import run_cv
from models import xgbm, catb, lgbm

X, y, Xt, _ = build("base")
NUMC = [c for c in X.columns if c not in CAT]
Xo = X.copy(); Xto = Xt.copy()  # ordinal-coded categoricals for sklearn trees
for c in CAT:
    cats = X[c].cat.categories
    Xo[c] = X[c].cat.codes; Xto[c] = pd.Categorical(Xt[c], categories=cats).codes


def sk(make, ordinal=True):
    def fp(Xa, ya, Xb, yb, Xte):
        m = make().fit(Xa, ya)
        return m.predict_proba(Xb)[:, 1], m.predict_proba(Xte)[:, 1]
    return fp


for name in sys.argv[1:]:
    if name == "lr":
        mk = lambda: make_pipeline(ColumnTransformer([("n", StandardScaler(), NUMC),
                                                      ("c", OneHotEncoder(handle_unknown="ignore"), CAT)]),
                                   LogisticRegression(C=1.0, max_iter=2000))
        run_cv("base_lr", sk(mk), X, y, Xt, verbose=False)
    elif name == "lr_balanced":
        mk = lambda: make_pipeline(ColumnTransformer([("n", StandardScaler(), NUMC),
                                                      ("c", OneHotEncoder(handle_unknown="ignore"), CAT)]),
                                   LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        run_cv("base_lr_balanced", sk(mk), X, y, Xt, verbose=False)
    elif name == "hgb":
        mk = lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=8, max_iter=1500,
                                                    min_samples_leaf=200, l2_regularization=1.0,
                                                    categorical_features=[Xo.columns.get_loc(c) for c in CAT],
                                                    early_stopping=True, validation_fraction=0.1,
                                                    n_iter_no_change=100, scoring="roc_auc", random_state=42)
        run_cv("base_hgb", sk(mk), Xo, y, Xto, verbose=False)
    elif name == "rf":
        mk = lambda: RandomForestClassifier(500, min_samples_leaf=100, max_features=0.3, n_jobs=20, random_state=42)
        run_cv("base_rf", sk(mk), Xo, y, Xto, verbose=False)
    elif name == "et":
        mk = lambda: ExtraTreesClassifier(500, min_samples_leaf=50, max_features=0.5, n_jobs=20, random_state=42)
        run_cv("base_et", sk(mk), Xo, y, Xto, verbose=False)
    elif name == "xgb":
        run_cv("base_xgb", xgbm(dict(max_depth=3)), X, y, Xt, verbose=False)
    elif name == "cat":
        run_cv("base_cat", catb(dict(depth=4)), X, y, Xt, verbose=False)
    elif name == "lgb_balanced":  # class-weighting experiment (same model as lgb_nl4_mcs200)
        run_cv("base_lgb_nl4_balanced", lgbm(dict(num_leaves=4, min_child_samples=200, learning_rate=0.05,
                                                  colsample_bytree=0.7, is_unbalance=True)), X, y, Xt, verbose=False)
