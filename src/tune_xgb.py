"""Optuna search for XGBoost-from-GLM on in-fold encodings.
Objective = AUC on folds 0 and 1 (scored fold untouched by early stopping, which uses an
inner 10% holdout). Best params saved to reports/best_xgb_params.json."""
import lightgbm  # noqa: F401
import json, os
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import REPORT_DIR
from cv import folds
from te import build_all
from train_te import fold_data, glm_fit

raw, codes, names, y, n, _ = build_all()
R, Rt = raw.values[:n].astype(np.float32), raw.values[n:].astype(np.float32)
FOLDS = []
for k, (a, b) in enumerate(folds(y)[:2]):
    Ttr, Tva, Tte = fold_data(k, a, b, raw, codes, y, n, False, 0)
    g = glm_fit(Ttr, y[a], [Tva])
    ia, ib = train_test_split(np.arange(len(a)), test_size=0.1, stratify=y[a], random_state=42)
    Xtr, Xva = np.hstack([R[a], Ttr]), np.hstack([R[b], Tva])
    FOLDS.append(dict(
        d1=xgb.DMatrix(Xtr[ia], y[a][ia], base_margin=g[0][ia]),
        d2=xgb.DMatrix(Xtr[ib], y[a][ib], base_margin=g[0][ib]),
        dv=xgb.DMatrix(Xva, base_margin=g[1]), yv=y[b]))


def objective(t):
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda",
             learning_rate=0.05, seed=42,
             max_depth=t.suggest_int("max_depth", 1, 5),
             min_child_weight=t.suggest_float("min_child_weight", 1, 300, log=True),
             subsample=t.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 0.9),
             reg_lambda=t.suggest_float("reg_lambda", 0.1, 100, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
             gamma=t.suggest_float("gamma", 1e-3, 5, log=True),
             max_bin=t.suggest_categorical("max_bin", [64, 128, 256, 512]))
    s = []
    for f in FOLDS:
        m = xgb.train(p, f["d1"], 10000, evals=[(f["d2"], "v")], early_stopping_rounds=200, verbose_eval=False)
        s.append(roc_auc_score(f["yv"], m.predict(f["dv"], iteration_range=(0, m.best_iteration + 1),
                                                  output_margin=True)))
        t.set_user_attr(f"iters{len(s)}", m.best_iteration)
    return float(np.mean(s))


if __name__ == "__main__":
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    st.enqueue_trial(dict(max_depth=2, min_child_weight=10, subsample=0.8, colsample_bytree=0.5,
                          reg_lambda=5.0, reg_alpha=0.001, gamma=0.001, max_bin=256))
    st.optimize(objective, n_trials=int(os.environ.get("TRIALS", 40)), show_progress_bar=False)
    print("baseline trial:", st.trials[0].value, "best:", st.best_value, st.best_params)
    with open(os.path.join(REPORT_DIR, "best_xgb_params.json"), "w") as f:
        json.dump(dict(best_value=st.best_value, params=st.best_params, baseline=st.trials[0].value,
                       n_trials=len(st.trials)), f, indent=2)
    st.trials_dataframe().to_csv(os.path.join(REPORT_DIR, "optuna_xgb_trials.csv"), index=False)
