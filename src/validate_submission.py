"""Structural checks for a submission file. usage: python validate_submission.py ../submission.csv"""
import os, sys
import numpy as np
import pandas as pd

from config import DATA_DIR

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(DATA_DIR), "submission.csv")
sub = pd.read_csv(path)
ss = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
checks = {
    "columns == ['id','is_fraud']": list(sub.columns) == ["id", "is_fraud"],
    "rows == 299,984": len(sub) == 299_984 == len(ss),
    "no missing values": not sub.isna().any().any(),
    "no duplicate ids": not sub["id"].duplicated().any(),
    "id order matches sample_submission": np.array_equal(sub["id"].values, ss["id"].values),
    "probabilities within [0,1]": bool(sub["is_fraud"].between(0, 1).all()),
    "predictions not constant": sub["is_fraud"].nunique() > 1000,
}
for k, v in checks.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
print(sub["is_fraud"].describe().to_string())
sys.exit(0 if all(checks.values()) else 1)
