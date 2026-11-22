"""Import Kaggle-GPU outputs (kg_*_oof.npy / kg_*_test.npy) into oof/ and build sub11+ on top of
sub02's recipe (logistic stack of all members on logit-ranks, scored by nested CV). CPU only.

usage: python import_kaggle.py <folder with the downloaded .npy files> [more folders...]
"""
import lightgbm  # noqa: F401
import glob, os, shutil, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import OOF_DIR, REPORT_DIR, SUB_DIR, DATA_DIR
from ensemble import load_preds, stacker, nested_auc
from features import load

tr, te = load(); y = tr["is_fraud"].values.astype(int); ids = te["id"].values
ss = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

new = []
for folder in sys.argv[1:]:
    for f in glob.glob(os.path.join(folder, "**", "test_ids.npy"), recursive=True):
        assert np.array_equal(np.load(f), ids), f"test id order differs in {f}"
    for f in sorted(glob.glob(os.path.join(folder, "**", "kg_*_oof.npy"), recursive=True)):
        name = os.path.basename(f)[:-8]; ft = f[:-8] + "_test.npy"
        o, t = np.load(f), np.load(ft)
        if len(o) != len(y) or len(t) != len(ids) or not np.isfinite(o).all() or not np.isfinite(t).all():
            print(f"SKIP {name}: wrong length or non-finite values"); continue
        auc = roc_auc_score(y, o)
        if auc < 0.90:
            print(f"SKIP {name}: OOF AUC {auc:.5f} looks broken"); continue
        shutil.copy(f, os.path.join(OOF_DIR, f"{name}_oof.npy")); shutil.copy(ft, os.path.join(OOF_DIR, f"{name}_test.npy"))
        new.append(name); print(f"imported {name:28s} OOF AUC {auc:.6f}")
if not new:
    sys.exit("no usable kg_* files found")

BASE = open(os.path.join(REPORT_DIR, "final_members.txt")).read().split()   # sub02's 22 members
sig = lambda z: 1 / (1 + np.exp(-z))
variants = [("sub11_sub02_plus_kaggle_C0.1", BASE + new, 0.1),
            ("sub12_sub02_plus_kaggle_C0.01", BASE + new, 0.01),
            ("sub13_kaggle_plus_glm_bags", new + ["te_lgb_glm_s42_t0", "te_lgb_glm_s42_t1", "te_lgb_glm_s42_t2",
                                                  "lgb_nl4_mcs200"], 0.1)]
O, _ = load_preds(BASE)
print(f"\nreference sub02 (22 members, C=0.1) nested CV: {nested_auc(O, y, 0.1):.6f}")
for name, mem, C in variants:
    O, T = load_preds(mem)
    na = nested_auc(O, y, C); m = stacker(C).fit(O, y)
    w = dict(zip(mem, np.round(m.coef_[0], 3)))
    sub = pd.DataFrame({"id": ids, "is_fraud": sig(m.decision_function(T))})
    assert np.array_equal(sub.id.values, ss.id.values) and sub.is_fraud.between(0, 1).all() and len(sub) == 299_984
    sub.to_csv(os.path.join(SUB_DIR, name + ".csv"), index=False)
    print(f"{name:34s} nested CV {na:.6f}   new-member weights: "
          + ", ".join(f"{k}={v}" for k, v in w.items() if k in new))
