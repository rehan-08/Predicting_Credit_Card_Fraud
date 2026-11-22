"""Exploratory data analysis: prints stats and saves plots to reports/eda/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import DATA_DIR, REPORT_DIR, TARGET

OUT = os.path.join(REPORT_DIR, "eda")
os.makedirs(OUT, exist_ok=True)

tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print("train", tr.shape, "test", te.shape)
print("dtypes:\n", tr.dtypes.value_counts())
print("missing train:", tr.isna().sum().sum(), "test:", te.isna().sum().sum())
feats = [c for c in tr.columns if c not in ("id", TARGET)]
print("dup rows (features only) train:", tr.duplicated(feats).sum())
both = pd.concat([tr[feats], te[feats]])
print("dup rows train+test:", both.duplicated().sum())
print("target mean:", tr[TARGET].mean(), "values:", tr[TARGET].unique())

num = [c for c in feats if tr[c].dtype != object and tr[c].dtype != bool]
cat = [c for c in feats if c not in num]
print("\nNumeric:", num)
print("Categorical/bool:", cat)

print("\n=== numeric describe ===")
print(tr[num].describe().T.round(3).to_string())

print("\n=== univariate AUC (numeric) & nunique ===")
rows = []
for c in num:
    a = roc_auc_score(tr[TARGET], tr[c])
    rows.append((c, tr[c].nunique(), te[c].nunique(), round(max(a, 1 - a), 4),
                 round(tr[c].mean(), 3), round(te[c].mean(), 3)))
print(pd.DataFrame(rows, columns=["feat", "nuniq_tr", "nuniq_te", "auc", "mean_tr", "mean_te"])
      .sort_values("auc", ascending=False).to_string(index=False))

print("\n=== categorical fraud rates ===")
for c in cat:
    g = tr.groupby(c)[TARGET].agg(["mean", "size"])
    g["test_size"] = te[c].value_counts()
    print(f"\n{c}\n", g.round(4).to_string())

# id artifact check
print("\n=== id check ===")
print("id AUC:", roc_auc_score(tr[TARGET], tr["id"]))
tr["id_bin"] = pd.qcut(tr["id"], 10, labels=False)
print(tr.groupby("id_bin")[TARGET].mean().round(4).to_string())
print("test id range:", te.id.min(), te.id.max())

print("\n=== numeric correlations (|r|>0.3) ===")
cm = tr[num + [TARGET]].corr()
pairs = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack()
print(pairs[pairs.abs() > 0.3].sort_values().to_string())

# Plots
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
for ax, c in zip(axes.ravel(), num):
    for t, col in [(0, "tab:blue"), (1, "tab:red")]:
        v = tr.loc[tr[TARGET] == t, c]
        ax.hist(v, bins=60, density=True, alpha=.5, color=col, label=f"fraud={t}")
    ax.set_title(c); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "numeric_dists.png"), dpi=80); plt.close()

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for ax, c in zip(axes.ravel(), cat):
    tr.groupby(c)[TARGET].mean().sort_values().plot.barh(ax=ax, color="tab:red")
    ax.set_title(f"fraud rate by {c}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "categorical_fraud_rates.png"), dpi=80); plt.close()

plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(cm)), cm.columns, rotation=90); plt.yticks(range(len(cm)), cm.columns)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "correlation.png"), dpi=80); plt.close()

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, c in zip(axes, ["amount_usd", "merchant_risk_score", "cvv_retry_count", "velocity_score"]):
    b = pd.qcut(tr[c].rank(method="first"), 20, labels=False)
    tr.groupby(b)[TARGET].mean().plot(ax=ax, marker="o"); ax.set_title(f"fraud rate vs {c} (ventile)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "key_relationships.png"), dpi=80); plt.close()
print("plots saved to", OUT)
