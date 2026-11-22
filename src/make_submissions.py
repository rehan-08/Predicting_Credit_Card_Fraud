"""Writes 10 submission variants to submissions/ with their CV score and a summary table.
Stacks are scored by nested CV; fixed blends/single models by plain OOF AUC (no fitted weights)."""
import lightgbm  # noqa: F401
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import REPORT_DIR, SUB_DIR
from ensemble import load_preds, stacker, nested_auc
from features import load

tr, te = load(); y = tr["is_fraud"].values.astype(int); ids = te["id"].values
ss = pd.read_csv(os.path.join(os.path.dirname(REPORT_DIR), "data", "sample_submission.csv"))
ALL = open(os.path.join(REPORT_DIR, "final_members.txt")).read().split()
TE = [m for m in ALL if m.startswith("te_")]
LGBG = ["te_lgb_glm_s42_t0", "te_lgb_glm_s42_t1", "te_lgb_glm_s42_t2"]
XGBG = ["te_xgb_glm_s42_t0_tuned", "te_xgb_glm_s42_t1_tuned", "te_xgb_glm_s42_t2_tuned"]
sig = lambda z: 1 / (1 + np.exp(-z))


def stack(members, C):
    O, T = load_preds(members)
    return nested_auc(O, y, C), sig(stacker(C).fit(O, y).decision_function(T)), "nested-CV stack"


def mean(members):
    O, T = load_preds(members)
    return roc_auc_score(y, O.mean(1)), sig(T.mean(1)), "OOF, equal weights"


V = [
    ("sub01_stack_all_C0.01", lambda: stack(ALL, 0.01)),
    ("sub02_stack_all_C0.1", lambda: stack(ALL, 0.1)),
    ("sub03_stack_all_C0.003", lambda: stack(ALL, 0.003)),
    ("sub04_stack_te_only", lambda: stack(TE, 0.01)),
    ("sub05_stack_glm_bags_cat_rawlgb", lambda: stack(LGBG + XGBG + ["te_cat_s42_t0", "lgb_nl4_mcs200"], 0.01)),
    ("sub06_mean_lgb_xgb_glm_bags", lambda: mean(LGBG + XGBG)),
    ("sub07_mean_xgb_glm_bag", lambda: mean(XGBG)),
    ("sub08_mean_lgb_glm_bag", lambda: mean(LGBG)),
    ("sub09_mean_top_te_models", lambda: mean(LGBG + XGBG + ["te_lgb_glm_s42_t0_m80", "te_xgb_glm_pairs_s42_t0",
                                                               "te_lgb_s42_t0", "te_cat_s42_t0"])),
    ("sub10_single_lgb_glm", lambda: mean(["te_lgb_glm_s42_t0"])),
]
rows = []
for name, fn in V:
    auc, p, how = fn()
    sub = pd.DataFrame({"id": ids, "is_fraud": p})
    ok = (list(sub.columns) == ["id", "is_fraud"] and len(sub) == 299_984 and not sub.isna().any().any()
          and not sub.id.duplicated().any() and np.array_equal(sub.id.values, ss.id.values)
          and sub.is_fraud.between(0, 1).all())
    sub.to_csv(os.path.join(SUB_DIR, name + ".csv"), index=False)
    rows.append(dict(file=name + ".csv", cv_auc=round(auc, 5), scored_by=how, checks="PASS" if ok else "FAIL"))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(SUB_DIR, "README_submissions.csv"), index=False)
print(df.to_string(index=False))
