"""Per-value target encoding of raw values and 'digit keys', fitted strictly in-fold.

Keys are factorized over train+test feature values (no target). Encodings are
computed only from the labels of the rows a model trains on:
  * training rows get inner-K-fold out-of-fold encodings,
  * validation/test rows get encodings fitted on all training rows.
"""
import lightgbm  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from features import load, NUM, CAT, BOOL

CONT = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
        "velocity_score", "merchant_risk_score"]
TWO_DEC = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]
PAIRS = [("merchant_category", "merchant_risk_score"), ("is_foreign_transaction", "ip_country_mismatch"),
         ("auth_method", "cvv_retry_count"), ("merchant_category", "auth_method"),
         ("channel", "device_type"), ("auth_method", "channel"), ("txn_count_last_24h", "cvv_retry_count"),
         ("prior_disputes", "cvv_retry_count"), ("merchant_category", "is_new_merchant")]


def key_frame(full, pairs=False):
    k = {}
    for c in NUM + CAT + BOOL:
        k[c] = full[c].astype(str)
    for c in CONT:
        v = np.round(full[c].values * 100).astype(np.int64)
        ip = v // 100
        k[f"{c}__int"] = ip
        k[f"{c}__int10"] = ip // 10
        k[f"{c}__tenth"] = (v // 10) % 10
        k[f"{c}__lead"] = pd.Series(ip).astype(str).str[0].values
        k[f"{c}__ndig"] = pd.Series(ip).astype(str).str.len().values
        if c in TWO_DEC:
            k[f"{c}__cent"] = v % 10
            k[f"{c}__frac"] = v % 100
    if pairs:
        for a, b in PAIRS:
            k[f"{a}__x__{b}"] = full[a].astype(str) + "|" + full[b].astype(str)
    K = pd.DataFrame(k)
    codes = np.column_stack([pd.factorize(K[c])[0] for c in K.columns]).astype(np.int32)
    return codes, list(K.columns)


def _enc(codes_fit, y_fit, codes_apply, prior, m):
    out = np.empty(codes_apply.shape, dtype=np.float32)
    for j in range(codes_fit.shape[1]):
        nk = max(codes_fit[:, j].max(), codes_apply[:, j].max()) + 1
        s = np.bincount(codes_fit[:, j], weights=y_fit, minlength=nk)
        n = np.bincount(codes_fit[:, j], minlength=nk)
        p = (s + m * prior) / (n + m)
        out[:, j] = np.log(p / (1 - p))[codes_apply[:, j]]
    return out


def fold_encode(C_tr, y_tr, C_apply_list, m=20, inner=5, seed=0):
    """Returns (TE for training rows via inner OOF, [TE for each apply matrix])."""
    prior = y_tr.mean()
    tr_te = np.empty(C_tr.shape, dtype=np.float32)
    for a, b in StratifiedKFold(inner, shuffle=True, random_state=seed).split(C_tr, y_tr):
        tr_te[b] = _enc(C_tr[a], y_tr[a], C_tr[b], y_tr[a].mean(), m)
    return tr_te, [_enc(C_tr, y_tr, C, prior, m) for C in C_apply_list]


def count_frame(full):
    return pd.DataFrame({f"cnt_{c}": full[c].map(full[c].value_counts()).values
                         for c in ["amount_usd", "account_balance_usd", "hours_since_last_txn",
                                   "distance_from_home_km", "velocity_score", "merchant_risk_score"]})


def build_all(pairs=False):
    """Raw numeric frame, key codes, y, test ids (train rows first, then test)."""
    tr, te = load()
    y = tr["is_fraud"].values.astype(int)
    full = pd.concat([tr.drop(columns="is_fraud"), te], ignore_index=True)
    raw = full[NUM].copy()
    for c in BOOL:
        raw[c] = full[c].astype(int)
    for c in CAT:
        raw[c] = pd.factorize(full[c], sort=True)[0]
    for c in TWO_DEC:
        raw[f"{c}_cents"] = np.round(full[c].values * 100).astype(np.int64) % 100
    raw = pd.concat([raw, count_frame(full)], axis=1)
    codes, names = key_frame(full, pairs)
    return raw, codes, names, y, len(tr), te["id"].values
