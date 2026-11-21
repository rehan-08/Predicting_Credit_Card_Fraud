# =====================================================================================
# S6E11 – Part 1 (Kaggle GPU notebook, accelerator "GPU T4 x2")
#   * in-fold per-value target encoding + digit keys (same recipe as the local pipeline)
#   * XGBoost boosted from a GLM logit, Optuna-tuned params, 5 encoding seeds   (GPU 0)
#   * CatBoost boosted from the GLM logit, 2 encoding seeds                     (GPU 0)
#   * Neural nets: per-value embeddings + encodings + raw, plain and GLM-residual (GPU 1)
# Folds: StratifiedKFold(5, shuffle=True, random_state=42) on train.csv row order
#        -> identical to the local pipeline, so OOF files can be stacked together.
# Output: /kaggle/working/out/<model>_oof.npy and <model>_test.npy (+ log.jsonl)
# Early stopping / model selection always uses a 10% inner holdout of the training fold.
# Smoke test anywhere:  SMOKE=1 python s6e11_part1_trees_nn.py <data_dir>
# =====================================================================================
import torch  # first: avoids a Windows DLL clash with sklearn/xgboost (harmless on Kaggle)
import torch.nn as nn
import os, sys, json, time, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SMOKE = os.environ.get("SMOKE") == "1"
CANDIDATES = [a for a in sys.argv[1:2]] + glob.glob("/kaggle/input/*s6e11*") + glob.glob("/kaggle/input/*")
DATA = next(d for d in CANDIDATES if os.path.exists(os.path.join(d, "train.csv")))
OUT = os.environ.get("OUT", "/kaggle/working/out" if os.path.exists("/kaggle/working") else "out_part1")
os.makedirs(OUT, exist_ok=True)
TE_SEEDS_XGB = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
TE_SEEDS_CAT = [0] if SMOKE else [0, 1]
NN_SEEDS = [0] if SMOKE else [0, 1]
print("data:", DATA, "out:", OUT, "smoke:", SMOKE, flush=True)

NGPU = torch.cuda.device_count()
DEV_TREE = "cuda:0" if NGPU else "cpu"
DEV_NN = f"cuda:{1 if NGPU > 1 else 0}" if NGPU else "cpu"
print("gpus:", NGPU, [torch.cuda.get_device_name(i) for i in range(NGPU)], flush=True)

# ------------------------------------------------------------------ data & keys
NUM = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
       "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
       "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]
CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
        "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
CONT = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
        "velocity_score", "merchant_risk_score"]
TWO_DEC = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]

tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
y_full = tr["is_fraud"].values.astype(int)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(tr)), y_full))
if SMOKE:  # keep fold structure honest: subsample rows, re-split
    keep = np.sort(np.random.default_rng(0).choice(len(tr), 30000, replace=False))
    tr = tr.iloc[keep].reset_index(drop=True); te = te.iloc[:5000].reset_index(drop=True)
    y_full = tr["is_fraud"].values.astype(int)
    FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(tr)), y_full))
y = y_full; n = len(tr)
full = pd.concat([tr.drop(columns="is_fraud"), te], ignore_index=True)

k = {}
for c in NUM + CAT + BOOL:
    k[c] = full[c].astype(str)
for c in CONT:
    v = np.round(full[c].values * 100).astype(np.int64); ip = v // 100
    k[f"{c}__int"] = ip; k[f"{c}__int10"] = ip // 10; k[f"{c}__tenth"] = (v // 10) % 10
    k[f"{c}__lead"] = pd.Series(ip).astype(str).str[0].values
    k[f"{c}__ndig"] = pd.Series(ip).astype(str).str.len().values
    if c in TWO_DEC:
        k[f"{c}__cent"] = v % 10; k[f"{c}__frac"] = v % 100
K = pd.DataFrame(k)
KEYS = list(K.columns)
CODES = np.column_stack([pd.factorize(K[c])[0] for c in KEYS]).astype(np.int32)
del K, k

raw = full[NUM].copy()
for c in BOOL: raw[c] = full[c].astype(int)
for c in CAT: raw[c] = pd.factorize(full[c], sort=True)[0]
for c in TWO_DEC: raw[f"{c}_cents"] = np.round(full[c].values * 100).astype(np.int64) % 100
for c in CONT: raw[f"cnt_{c}"] = full[c].map(full[c].value_counts()).values
RAW = raw.values.astype(np.float32); del raw
R, Rt = RAW[:n], RAW[n:]


def _enc(cf, yf, ca, prior, m=20):
    out = np.empty(ca.shape, dtype=np.float32)
    for j in range(cf.shape[1]):
        nk = max(cf[:, j].max(), ca[:, j].max()) + 1
        s = np.bincount(cf[:, j], weights=yf, minlength=nk); c = np.bincount(cf[:, j], minlength=nk)
        p = (s + m * prior) / (c + m)
        out[:, j] = np.log(p / (1 - p))[ca[:, j]]
    return out


def fold_encode(Ctr, ytr, applies, seed):
    trte = np.empty(Ctr.shape, dtype=np.float32)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(Ctr, ytr):
        trte[b] = _enc(Ctr[a], ytr[a], Ctr[b], ytr[a].mean())
    return trte, [_enc(Ctr, ytr, C, ytr.mean()) for C in applies]


def glm_fit(Ttr, ytr, Tlist):
    sc = StandardScaler().fit(Ttr)
    m = LogisticRegression(C=1.0, solver="newton-cholesky", max_iter=100).fit(sc.transform(Ttr), ytr)
    return [m.decision_function(sc.transform(T)) for T in [Ttr] + Tlist]


# ------------------------------------------------------------------ models
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

XGB_P = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device=DEV_TREE,
             learning_rate=0.02, max_depth=3, min_child_weight=4.235, subsample=0.745,
             colsample_bytree=0.875, reg_lambda=2.506, reg_alpha=0.0116, gamma=0.0123, max_bin=128)


def run_xgb(Xtr, ytr, Xva, Xte, init, seed):
    ia, ib = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=seed)
    p = {**XGB_P, "seed": seed}
    d1 = xgb.DMatrix(Xtr[ia], ytr[ia], base_margin=init[0][ia]); d2 = xgb.DMatrix(Xtr[ib], ytr[ib], base_margin=init[0][ib])
    m = xgb.train(p, d1, 200 if SMOKE else 20000, evals=[(d2, "v")], early_stopping_rounds=300, verbose_eval=False)
    m = xgb.train(p, xgb.DMatrix(Xtr, ytr, base_margin=init[0]), int((m.best_iteration + 1) * 1.1))
    return (m.predict(xgb.DMatrix(Xva, base_margin=init[1]), output_margin=True),
            m.predict(xgb.DMatrix(Xte, base_margin=init[2]), output_margin=True))


def run_cat(Xtr, ytr, Xva, Xte, init, seed):
    ia, ib = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=seed)
    p = dict(learning_rate=0.05, depth=4, l2_leaf_reg=10, eval_metric="AUC", random_seed=seed, verbose=0,
             task_type="GPU" if NGPU else "CPU", devices="0", border_count=254)
    m = CatBoostClassifier(iterations=200 if SMOKE else 20000, od_type="Iter", od_wait=300, **p)
    m.fit(Pool(Xtr[ia], ytr[ia], baseline=init[0][ia]), eval_set=Pool(Xtr[ib], ytr[ib], baseline=init[0][ib]))
    m = CatBoostClassifier(iterations=int((m.get_best_iteration() + 1) * 1.1), **p)
    m.fit(Pool(Xtr, ytr, baseline=init[0]))
    return (m.predict(Pool(Xva, baseline=init[1]), prediction_type="RawFormulaVal"),
            m.predict(Pool(Xte, baseline=init[2]), prediction_type="RawFormulaVal"))



class Net(nn.Module):
    def __init__(self, cards, n_dense, emb=12, h=512, drop=0.15):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb, c)) for c in cards])
        d = sum(min(emb, c) for c in cards) + n_dense
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Dropout(drop), nn.Linear(h, h // 2), nn.SiLU(),
                                 nn.Dropout(drop), nn.Linear(h // 2, 1))

    def forward(self, xc, xd):
        e = [m(xc[:, i]) for i, m in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [xd], 1)).squeeze(1)


def nn_codes(Ctr_full_idx):
    """Per-key codes for embeddings: values seen < 10 times in this training fold -> 0."""
    out = np.zeros_like(CODES); cards = []
    for j in range(CODES.shape[1]):
        c = CODES[:, j]; cnt = np.bincount(c[Ctr_full_idx], minlength=c.max() + 1)
        keep = cnt >= 10; remap = np.where(keep, np.cumsum(keep), 0)
        out[:, j] = remap[c]; cards.append(int(keep.sum()) + 1)
    return out, cards


def run_nn(a_idx, Dtr, ytr, Dva, Dte, init, residual, seed, epochs=4):
    torch.manual_seed(seed); np.random.seed(seed)
    codes, cards = nn_codes(a_idx)
    Ctr, Cva, Cte = codes[:n][a_idx], codes[:n][VA_IDX], codes[n:]
    sc = StandardScaler().fit(Dtr)
    Dtr, Dva, Dte = [np.nan_to_num(sc.transform(D)).astype(np.float32) for D in (Dtr, Dva, Dte)]
    ia, ib = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=seed)
    dev = torch.device(DEV_NN)
    T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt)
    off = [np.zeros(len(D), np.float32) for D in (Dtr, Dva, Dte)] if not residual else [i.astype(np.float32) for i in init]
    net = Net(cards, Dtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    bs = 2048; steps = epochs * int(np.ceil(len(ia) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()

    def predict(C, D, o):
        net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(D), 16384):
                out.append((net(T(C[s:s+16384], torch.long).to(dev), T(D[s:s+16384]).to(dev)).float().cpu().numpy()
                            + o[s:s+16384]))
        net.train(); return np.concatenate(out)

    best, best_state = -1, None
    yt = T(ytr[ia]).to(dev); Ct = T(Ctr[ia], torch.long).to(dev); Dt = T(Dtr[ia]).to(dev); Ot = T(off[0][ia]).to(dev)
    for ep in range(epochs):
        perm = torch.randperm(len(ia), device=dev)
        for s in range(0, len(ia), bs):
            i = perm[s:s+bs]
            loss = lossf(net(Ct[i], Dt[i]) + Ot[i], yt[i])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        auc = roc_auc_score(ytr[ib], predict(Ctr[ib], Dtr[ib], off[0][ib]))
        if auc > best:
            best = auc; best_state = {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}
    net.load_state_dict(best_state)
    return predict(Cva, Dva, off[1]), predict(Cte, Dte, off[2])


# ------------------------------------------------------------------ main loop
LOGF = os.path.join(OUT, "log.jsonl")
acc = {}  # name -> [oof, test, fold_count]


def add(name, b, pv, pt):
    if name not in acc: acc[name] = [np.zeros(n), np.zeros(len(te)), 0]
    acc[name][0][b] = pv; acc[name][1] += pt / 5; acc[name][2] += 1


t0 = time.time()
all_seeds = sorted(set(TE_SEEDS_XGB) | set(TE_SEEDS_CAT) | set(NN_SEEDS))
for s in all_seeds:
    for fk, (a, b) in enumerate(FOLDS):
        VA_IDX = b
        C = CODES[:n]
        Ttr, (Tva, Tte) = fold_encode(C[a], y[a], [C[b], CODES[n:]], seed=s)
        init = glm_fit(Ttr, y[a], [Tva, Tte])
        Xtr, Xva, Xte = np.hstack([R[a], Ttr]), np.hstack([R[b], Tva]), np.hstack([Rt, Tte])
        if s in TE_SEEDS_XGB:
            pv, pt = run_xgb(Xtr, y[a], Xva, Xte, init, seed=42 + s); add(f"kg_xgb_glm_t{s}", b, pv, pt)
        if s in TE_SEEDS_CAT:
            pv, pt = run_cat(Xtr, y[a], Xva, Xte, init, seed=42 + s); add(f"kg_cat_glm_t{s}", b, pv, pt)
        if s in NN_SEEDS:
            for resid in (False, True):
                pv, pt = run_nn(a, Xtr, y[a], Xva, Xte, init, resid, seed=42 + s, epochs=1 if SMOKE else 4)
                add(f"kg_nn{'_resid' if resid else ''}_t{s}", b, pv, pt)
        print(f"seed {s} fold {fk} done ({time.time()-t0:.0f}s)", flush=True)
    for name, (oof, tp, cnt) in acc.items():
        if cnt == 5 and not os.path.exists(os.path.join(OUT, f"{name}_oof.npy")):
            auc = roc_auc_score(y, oof)
            np.save(os.path.join(OUT, f"{name}_oof.npy"), oof); np.save(os.path.join(OUT, f"{name}_test.npy"), tp)
            with open(LOGF, "a") as f: f.write(json.dumps(dict(name=name, auc=round(auc, 6))) + "\n")
            print(f"RESULT {name} OOF AUC {auc:.6f}", flush=True)

np.save(os.path.join(OUT, "test_ids.npy"), te["id"].values)
print("all done", time.time() - t0)
