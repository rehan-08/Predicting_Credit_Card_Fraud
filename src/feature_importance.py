"""Feature importance on fold 0: LightGBM gain on raw+encoded features, plus the
GLM's standardized coefficients. Saves reports/feature_importance.csv and a plot."""
import lightgbm as lgb
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import REPORT_DIR
from cv import folds
from te import build_all
from train_te import fold_data

raw, codes, names, y, n, _ = build_all()
a, b = folds(y)[0]
Ttr, Tva, Tte = fold_data(0, a, b, raw, codes, y, n, False, 0)
cols = list(raw.columns) + [f"te:{c}" for c in names]
X = np.hstack([raw.values[:n][a].astype(np.float32), Ttr])
m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=3, min_child_samples=200,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1,
                   n_jobs=20, seed=42), lgb.Dataset(X, y[a].astype(float), feature_name=[
                       c.replace(":", "_") for c in cols]), 2000)
gain = pd.Series(m.feature_importance("gain"), index=cols)
sc = StandardScaler().fit(Ttr)
glm = LogisticRegression(C=1.0, solver="newton-cholesky").fit(sc.transform(Ttr), y[a])
coef = pd.Series(glm.coef_[0], index=[f"te:{c}" for c in names])
df = pd.DataFrame({"lgb_gain": gain, "lgb_gain_pct": 100 * gain / gain.sum(), "glm_std_coef": coef})
df = df.sort_values("lgb_gain", ascending=False)
df.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"))
print(df.head(30).round(4).to_string())

top = df.head(30)[::-1]
plt.figure(figsize=(9, 10))
plt.barh(top.index, top.lgb_gain_pct, color="tab:blue")
plt.xlabel("% of total gain (LightGBM, fold 0)"); plt.title("Top 30 features")
plt.tight_layout(); plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=90)
