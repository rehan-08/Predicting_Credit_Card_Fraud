"""Feature-set ablation with a fast LightGBM config. Usage: python exp_features.py base eng ..."""
import sys
from features import build
from cv import run_cv
from models import lgbm

for fs in sys.argv[1:]:
    X, y, Xt, _ = build(fs)
    fp = lgbm(dict(learning_rate=0.05))
    run_cv(f"lgb_fs-{fs}", fp, X, y, Xt, extra=dict(fs=fs))
    print("  best iters", fp.best_iters)
