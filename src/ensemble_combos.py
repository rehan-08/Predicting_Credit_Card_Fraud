"""Nested-CV AUC for the requested ensemble combinations (family = seed-bagged group)."""
import lightgbm  # noqa: F401
import numpy as np
from sklearn.metrics import roc_auc_score

from ensemble import load_preds, nested_auc
from features import load

FAM = {
    "LGB": ["te_lgb_glm_s42_t0", "te_lgb_glm_s42_t1", "te_lgb_glm_s42_t2"],
    "XGB": ["te_xgb_glm_s42_t0_tuned", "te_xgb_glm_s42_t1_tuned", "te_xgb_glm_s42_t2_tuned"],
    "CAT": ["te_cat_s42_t0"],
}
tr, _ = load(); y = tr["is_fraud"].values.astype(int)
P = {k: load_preds(v)[0].mean(1) for k, v in FAM.items()}  # seed-bag = mean of logit-ranks
for k, v in P.items():
    print(f"{k} (seed bag)            OOF AUC {roc_auc_score(y, v):.6f}")
for combo in [("LGB", "XGB"), ("LGB", "CAT"), ("XGB", "CAT"), ("LGB", "XGB", "CAT")]:
    O = np.column_stack([P[c] for c in combo])
    w = np.array([0.4, 0.3, 0.3]) if len(combo) == 3 else np.array([0.5, 0.5])
    print(f"{'+'.join(combo):18s} fixed-weight {roc_auc_score(y, O @ w):.6f}   "
          f"nested-CV LR stack {nested_auc(O, y):.6f}")
