"""Probe model capacity: shallow vs deep LightGBM."""
from features import build
from cv import run_cv
from models import lgbm
X, y, Xt, _ = build("base")
for nl, mcs, lr in [(4, 200, 0.05), (8, 500, 0.05), (15, 1000, 0.05)]:
    fp = lgbm(dict(num_leaves=nl, min_child_samples=mcs, learning_rate=lr, colsample_bytree=0.7))
    run_cv(f"lgb_nl{nl}_mcs{mcs}", fp, X, y, Xt, verbose=False, extra=dict(fs="base"))
    print("  iters", fp.best_iters, flush=True)
