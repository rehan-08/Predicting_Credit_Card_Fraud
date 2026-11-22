import lightgbm  # noqa: F401  must load before sklearn/pandas on Windows (OpenMP DLL clash -> segfault)
"""Feature engineering. All transforms are row-wise or fit on train+test feature
values only (no target), so they are leakage-free. Target encoding is done
inside CV folds in cv.py, never here."""
import os
import numpy as np
import pandas as pd

from config import DATA_DIR, TARGET

BOOL = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
        "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
CAT = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
NUM = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
       "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
       "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]


def load():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return tr, te


def _base(df):
    out = df[NUM].copy()
    for c in BOOL:
        out[c] = df[c].astype(int)
    for c in CAT:
        out[c] = df[c].astype("category")
    return out


def add_eng(out, df):
    """Domain features (transaction, risk, time, interactions)."""
    out["log_amount"] = np.log1p(df.amount_usd)
    out["amt_to_bal"] = df.amount_usd / (df.account_balance_usd + 1)
    out["amt_per_txn"] = df.amount_usd / (df.txn_count_last_24h + 1)
    out["txn_rate"] = df.txn_count_last_24h / (df.hours_since_last_txn + 0.1)
    out["risk_x_vel"] = df.merchant_risk_score * df.velocity_score
    out["flag_sum"] = df[BOOL].astype(int).sum(1)
    out["geo_mismatch"] = (df.is_foreign_transaction.astype(int) + df.ip_country_mismatch.astype(int)
                           + df.billing_shipping_mismatch.astype(int))
    out["no_auth"] = (df.auth_method == "No Authentication").astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * df.time_of_day_hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df.time_of_day_hour / 24)
    out["is_night"] = df.time_of_day_hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    out["is_weekend"] = df.day_of_week.isin([5, 6]).astype(int)
    out["mcat_card"] = (df.merchant_category + "_" + df.card_type).astype("category")
    out["auth_channel"] = (df.auth_method + "_" + df.channel).astype("category")
    out["device_vpn"] = (df.device_type + "_" + df.used_vpn.astype(str)).astype("category")
    out["foreign_ipmis"] = df.is_foreign_transaction.astype(int) * 2 + df.ip_country_mismatch.astype(int)
    out["newm_foreign"] = df.is_new_merchant.astype(int) * 2 + df.is_foreign_transaction.astype(int)
    return out


def build(fs="base"):
    """Return X_train, y, X_test for a named feature set."""
    tr, te = load()
    y = tr[TARGET].values.astype(int)
    full = pd.concat([tr.drop(columns=[TARGET]), te], ignore_index=True)
    X = _base(full)
    if "eng" in fs:
        X = add_eng(X, full)
    if "cnt" in fs:
        for c in NUM:
            X[f"cnt_{c}"] = full[c].map(full[c].value_counts()).values
    if "rnd" in fs:  # decimal-part artifacts of synthetic generation
        for c in ["amount_usd", "account_balance_usd", "velocity_score", "merchant_risk_score",
                  "hours_since_last_txn", "distance_from_home_km"]:
            X[f"frac_{c}"] = np.round((full[c] * 100) % 100).astype(int)
    if "catnum" in fs:  # numerics as categorical (handled via native cats)
        for c in ["merchant_risk_score", "velocity_score", "customer_age", "card_age_months"]:
            X[f"cat_{c}"] = full[c].astype(str).astype("category")
    n = len(tr)
    return X.iloc[:n].reset_index(drop=True), y, X.iloc[n:].reset_index(drop=True), te["id"].values
