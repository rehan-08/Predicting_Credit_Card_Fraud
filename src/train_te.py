"""Models on in-fold target encodings + digit keys.

usage: python train_te.py <model> [tag] [--pairs] [--seed S] [--te-seed S]
models: glm | lgb | lgb_glm | xgb | xgb_glm | cat
Early stopping uses a 10% inner holdout of the training fold, never the scored fold.
"""
import lightgbm as lgb
import argparse, json, os, time
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import OOF_DIR, ROOT, SEED
from cv import folds, LOG
from te import build_all, fold_encode

CACHE = os.path.join(ROOT, "cache")
os.makedirs(CACHE, exist_ok=True)


def fold_data(k, a, b, raw, codes, y, n, pairs, te_seed, m=20):
    tag = f"f{k}_p{int(pairs)}_s{te_seed}" + ("" if m == 20 else f"_m{m}")
    p = os.path.join(CACHE, tag + ".npz")
    if os.path.exists(p):
        z = np.load(p); return z["tr"], z["va"], z["te"]
    C, Ct = codes[:n], codes[n:]
    tr_te, (va_te, te_te) = fold_encode(C[a], y[a], [C[b], Ct], m=m, seed=te_seed)
    np.savez(p, tr=tr_te, va=va_te, te=te_te)
    return tr_te, va_te, te_te


def glm_fit(Ttr, ytr, Tlist):
    sc = StandardScaler().fit(Ttr)
    m = LogisticRegression(C=1.0, solver="newton-cholesky", max_iter=100)
    m.fit(sc.transform(Ttr), ytr)
    return [m.decision_function(sc.transform(T)) for T in [Ttr] + Tlist]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag", nargs="?", default="")
    ap.add_argument("--pairs", action="store_true"); ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--te-seed", type=int, default=0); ap.add_argument("--params", default="{}")
    ap.add_argument("--no-raw", action="store_true"); ap.add_argument("--m", type=int, default=20)
    args = ap.parse_args()
    P = json.loads(args.params)

    raw, codes, names, y, n, _ = build_all(args.pairs)
    R, Rt = raw.values[:n].astype(np.float32), raw.values[n:].astype(np.float32)
    oof = np.zeros(n); tp = np.zeros(len(Rt)); t0 = time.time(); iters = []
    for k, (a, b) in enumerate(folds(y)):
        Ttr, Tva, Tte = fold_data(k, a, b, raw, codes, y, n, args.pairs, args.te_seed, args.m)
        if args.model == "glm":
            _, pv, pt = glm_fit(Ttr, y[a], [Tva, Tte])
        else:
            if args.no_raw:
                Xtr, Xva, Xte = Ttr, Tva, Tte
            else:
                Xtr, Xva, Xte = (np.hstack([R[a], Ttr]), np.hstack([R[b], Tva]), np.hstack([Rt, Tte]))
            init = None
            if args.model.endswith("_glm"):
                init = glm_fit(Ttr, y[a], [Tva, Tte])
            ia, ib = train_test_split(np.arange(len(a)), test_size=0.1, stratify=y[a], random_state=args.seed)
            if args.model.startswith("lgb"):
                prm = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=3,
                           min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           reg_lambda=5.0, max_bin=255, verbose=-1, n_jobs=20, seed=args.seed)
                prm.update(P)
                d1 = lgb.Dataset(Xtr[ia], y[a][ia].astype(float), init_score=None if init is None else init[0][ia],
                                 free_raw_data=False)
                d2 = lgb.Dataset(Xtr[ib], y[a][ib].astype(float), init_score=None if init is None else init[0][ib],
                                 reference=d1)
                m = lgb.train(prm, d1, 20000, valid_sets=[d2], callbacks=[lgb.early_stopping(300, verbose=False)])
                bi = m.best_iteration; iters.append(bi)
                # refit on full training fold with the found number of rounds (x1.1)
                dfull = lgb.Dataset(Xtr, y[a].astype(float), init_score=None if init is None else init[0])
                m = lgb.train(prm, dfull, int(bi * 1.1))
                pv = m.predict(Xva, raw_score=True); pt = m.predict(Xte, raw_score=True)
                if init is not None:  # lightgbm does not store init_score in the model
                    pv = pv + init[1]; pt = pt + init[2]
            elif args.model.startswith("xgb"):
                prm = dict(learning_rate=0.03, max_depth=2, min_child_weight=10, subsample=0.8,
                           colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist", device="cuda",
                           objective="binary:logistic", eval_metric="auc", seed=args.seed)
                prm.update(P)
                bm = lambda X, idx, im: xgb.DMatrix(X[idx], label=y[a][idx] if idx is not None else None,
                                                     base_margin=None if im is None else im)
                d1 = xgb.DMatrix(Xtr[ia], y[a][ia], base_margin=None if init is None else init[0][ia])
                d2 = xgb.DMatrix(Xtr[ib], y[a][ib], base_margin=None if init is None else init[0][ib])
                m = xgb.train(prm, d1, 20000, evals=[(d2, "v")], early_stopping_rounds=300, verbose_eval=False)
                bi = m.best_iteration + 1; iters.append(bi)
                m = xgb.train(prm, xgb.DMatrix(Xtr, y[a], base_margin=None if init is None else init[0]),
                              int(bi * 1.1))
                pv = m.predict(xgb.DMatrix(Xva, base_margin=None if init is None else init[1]), output_margin=True)
                pt = m.predict(xgb.DMatrix(Xte, base_margin=None if init is None else init[2]), output_margin=True)
            elif args.model == "cat":
                from catboost import CatBoostClassifier
                prm = dict(learning_rate=0.05, depth=3, l2_leaf_reg=10, iterations=20000, eval_metric="AUC",
                           od_type="Iter", od_wait=300, task_type="GPU", verbose=0, random_seed=args.seed)
                prm.update(P)
                m = CatBoostClassifier(**prm)
                m.fit(Xtr[ia], y[a][ia], eval_set=(Xtr[ib], y[a][ib]))
                bi = m.get_best_iteration() + 1; iters.append(bi)
                prm.update(iterations=int(bi * 1.1)); prm.pop("od_type"); prm.pop("od_wait")
                m = CatBoostClassifier(**prm).fit(Xtr, y[a])
                pv = m.predict(Xva, prediction_type="RawFormulaVal"); pt = m.predict(Xte, prediction_type="RawFormulaVal")
        oof[b] = pv; tp += np.asarray(pt) / 5
        print(f"  fold {k} auc={roc_auc_score(y[b], pv):.5f} iters={iters[-1] if iters else '-'} "
              f"({time.time()-t0:.0f}s)", flush=True)
    name = f"te_{args.model}{'_pairs' if args.pairs else ''}{'_noraw' if args.no_raw else ''}" \
           f"_s{args.seed}_t{args.te_seed}{'' if args.m == 20 else f'_m{args.m}'}{args.tag}"
    auc = roc_auc_score(y, oof)
    res = dict(name=name, auc=round(auc, 6), params=P, iters=iters, secs=round(time.time() - t0))
    print("RESULT", json.dumps(res), flush=True)
    np.save(os.path.join(OOF_DIR, f"{name}_oof.npy"), oof)   # raw logits
    np.save(os.path.join(OOF_DIR, f"{name}_test.npy"), tp)
    with open(LOG, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
