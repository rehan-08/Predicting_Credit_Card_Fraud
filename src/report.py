"""Builds reports/model_comparison.md and threshold metrics for the final nested-CV stack."""
import lightgbm  # noqa: F401
import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_fscore_support,
                             confusion_matrix, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold

from config import REPORT_DIR, OOF_DIR
from ensemble import load_preds, stacker
from features import load

tr, _ = load(); y = tr["is_fraud"].values.astype(int)
members = open(os.path.join(REPORT_DIR, "final_members.txt")).read().split()
O, _ = load_preds(members)
p = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(O, y):
    p[b] = stacker().fit(O[a], y[a]).decision_function(O[b])
p = 1 / (1 + np.exp(-p))


def thr_metrics(pred, name):
    pr, rc, th = precision_recall_curve(y, pred)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = np.nanargmax(f1[:-1]); t = th[i]
    out = {}
    for label, cut in [("F1-optimal", t), ("0.5", 0.5), ("top 1.67% flagged", np.quantile(pred, 1 - y.mean()))]:
        yh = (pred >= cut).astype(int)
        P, R, F, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        out[label] = dict(threshold=float(round(cut, 5)), precision=round(P, 4), recall=round(R, 4), f1=round(F, 4),
                          TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    return dict(model=name, roc_auc=round(roc_auc_score(y, pred), 6), pr_auc=round(average_precision_score(y, pred), 6),
                thresholds=out)


res = [thr_metrics(p, "final stack (nested CV)")]
for nm in ["te_xgb_glm_s42_t0_tuned", "te_lgb_glm_s42_t0", "lgb_nl4_mcs200", "base_lr"]:
    o = np.load(os.path.join(OOF_DIR, f"{nm}_oof.npy"))
    res.append(thr_metrics(1 / (1 + np.exp(-o)) if o.min() < 0 or o.max() > 1 else o, nm))
with open(os.path.join(REPORT_DIR, "final_metrics.json"), "w") as f:
    json.dump(res, f, indent=2)
for r in res:
    t = r["thresholds"]["F1-optimal"]
    print(f"{r['model']:28s} ROC-AUC {r['roc_auc']:.5f}  PR-AUC {r['pr_auc']:.4f}  @F1-opt thr={t['threshold']}: "
          f"P={t['precision']} R={t['recall']} F1={t['f1']}  CM[TN,FP,FN,TP]=[{t['TN']},{t['FP']},{t['FN']},{t['TP']}]")

rows = [json.loads(l) for l in open(os.path.join(REPORT_DIR, "experiments.jsonl"))]
df = pd.DataFrame(rows).drop_duplicates("name", keep="last")
df = df[df.auc > 0.7].sort_values("auc", ascending=False)  # drops the buggy lgb_glm run (fixed & rerun)
with open(os.path.join(REPORT_DIR, "model_comparison.md"), "w") as f:
    f.write("| Model | 5-fold OOF ROC-AUC | PR-AUC |\n|---|---:|---:|\n")
    for _, r in df.iterrows():
        f.write(f"| {r['name']} | {r['auc']:.5f} | {'' if pd.isna(r.get('pr_auc')) else f'{r.pr_auc:.4f}'} |\n")
print(f"\n{len(df)} experiments written to reports/model_comparison.md")
