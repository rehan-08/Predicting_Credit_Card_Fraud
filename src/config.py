import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORT_DIR = os.path.join(ROOT, "reports")
OOF_DIR = os.path.join(ROOT, "oof")
SUB_DIR = os.path.join(ROOT, "submissions")
TARGET = "is_fraud"
SEED = 42
N_FOLDS = 5

for d in (REPORT_DIR, OOF_DIR, SUB_DIR):
    os.makedirs(d, exist_ok=True)
