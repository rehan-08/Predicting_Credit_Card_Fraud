# =====================================================================================
# S6E11 – Part 2 (Kaggle GPU notebook, accelerator "GPU T4 x2", Internet ON)
# GPT-2 sequence classifier over rows written as text. It reads the digits of every value,
# so it ranks rows differently from the tree models (useful diversity for the stack).
#
# Run in ONE notebook cell (uses both T4s, one process per GPU):
#     !python s6e11_part2_gpt2.py launch
# Single worker (what "launch" starts):  python s6e11_part2_gpt2.py worker --gpu 0 --folds 0,2,4
# Merge only (if a run was interrupted): python s6e11_part2_gpt2.py merge
# Smoke test on CPU (tiny random GPT-2, no download):  SMOKE=1 python s6e11_part2_gpt2.py launch <data_dir>
#
# Folds: StratifiedKFold(5, shuffle=True, random_state=42) – identical to the local pipeline.
# No early stopping / no selection on the scored fold: fixed 1 epoch, fixed schedule.
# Output: /kaggle/working/out/kg_gpt2_oof.npy, kg_gpt2_test.npy
# =====================================================================================
import os, sys, json, time, glob, subprocess
import numpy as np
import pandas as pd

SMOKE = os.environ.get("SMOKE") == "1"
MODEL = os.environ.get("LM_MODEL", "gpt2")          # "distilgpt2" is ~2x faster, slightly weaker
EPOCHS = float(os.environ.get("LM_EPOCHS", "1"))
BS, LR, MAXLEN = 64, 1e-4, 160
args = sys.argv[1:]
mode = args[0] if args else "launch"
pos = [a for a in args[1:] if not a.startswith("--") and not a[0].isdigit()]
CANDIDATES = pos[:1] + glob.glob("/kaggle/input/*s6e11*") + glob.glob("/kaggle/input/*")
DATA = next(d for d in CANDIDATES if os.path.exists(os.path.join(d, "train.csv")))
OUT = os.environ.get("OUT", "/kaggle/working/out" if os.path.exists("/kaggle/working") else "out_part2")
os.makedirs(OUT, exist_ok=True)


def load():
    from sklearn.model_selection import StratifiedKFold
    tr = pd.read_csv(os.path.join(DATA, "train.csv")); te = pd.read_csv(os.path.join(DATA, "test.csv"))
    if SMOKE:
        tr = tr.sample(6000, random_state=0).reset_index(drop=True); te = te.iloc[:1000].reset_index(drop=True)
    y = tr["is_fraud"].values.astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(tr)), y))
    return tr, te, y, folds


SHORT = {"amount_usd": "amt", "hours_since_last_txn": "hrs", "txn_count_last_24h": "n24", "distance_from_home_km": "dist",
         "card_age_months": "card", "customer_age": "age", "account_balance_usd": "bal", "cvv_retry_count": "cvv",
         "velocity_score": "vel", "time_of_day_hour": "hour", "day_of_week": "dow", "merchant_risk_score": "risk",
         "prior_disputes": "disp", "merchant_category": "mcat", "card_type": "ctype", "auth_method": "auth",
         "channel": "chan", "device_type": "dev", "is_foreign_transaction": "foreign", "is_new_merchant": "newm",
         "used_vpn": "vpn", "ip_country_mismatch": "ipmis", "billing_shipping_mismatch": "bsmis",
         "is_ai_generated_scam_attempt": "ai"}


def to_text(df):
    parts = []
    for c, s in SHORT.items():
        v = df[c]
        if v.dtype == bool: v = v.map({True: "y", False: "n"})
        parts.append(s + " " + v.astype(str))
    out = parts[0]
    for p in parts[1:]: out = out + " | " + p
    return out.tolist()


def worker(gpu, folds_to_run):
    import torch
    from transformers import AutoTokenizer, GPT2ForSequenceClassification, GPT2Config, get_linear_schedule_with_warmup
    from sklearn.metrics import roc_auc_score
    # launch sets CUDA_VISIBLE_DEVICES=<gpu>, so the worker's GPU is always cuda:0
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tr, te, y, folds = load()
    try:
        tok = AutoTokenizer.from_pretrained("gpt2" if SMOKE else MODEL)
    except Exception as e:
        raise SystemExit(f"Tokenizer download failed ({e}). Turn Internet ON in the notebook settings.")
    # right padding: GPT2ForSequenceClassification reads the last non-pad token (works in all versions)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    enc = lambda texts: tok(texts, truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="np")
    Xtr = enc(to_text(tr)); Xte = enc(to_text(te))
    L = int(max(Xtr["attention_mask"].sum(1).max(), Xte["attention_mask"].sum(1).max()))
    Xtr = {k: v[:, :L] for k, v in Xtr.items()}; Xte = {k: v[:, :L] for k, v in Xte.items()}
    print(f"[gpu{gpu}] seq len {L}", flush=True)

    def predict(model, X, idx):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            for s in range(0, len(idx), 512):
                i = idx[s:s+512]
                lo = model(input_ids=torch.as_tensor(X["input_ids"][i]).to(dev),
                           attention_mask=torch.as_tensor(X["attention_mask"][i]).to(dev)).logits
                out.append(lo[:, 1].float().cpu().numpy() - lo[:, 0].float().cpu().numpy())
        model.train(); return np.concatenate(out)

    for fk in folds_to_run:
        a, b = folds[fk]; t0 = time.time()
        torch.manual_seed(42 + fk)
        if SMOKE:
            model = GPT2ForSequenceClassification(GPT2Config(n_layer=2, n_head=2, n_embd=64, num_labels=2,
                                                             pad_token_id=tok.eos_token_id))
        else:
            model = GPT2ForSequenceClassification.from_pretrained(MODEL, num_labels=2)
            model.config.pad_token_id = tok.eos_token_id
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = int(np.ceil(len(a) / BS) * EPOCHS)
        sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
        rng = np.random.default_rng(fk); step = 0
        while step < steps:
            perm = rng.permutation(a)
            for s in range(0, len(perm), BS):
                if step >= steps: break
                i = perm[s:s+BS]
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    loss = model(input_ids=torch.as_tensor(Xtr["input_ids"][i]).to(dev),
                                 attention_mask=torch.as_tensor(Xtr["attention_mask"][i]).to(dev),
                                 labels=torch.as_tensor(y[i], dtype=torch.long).to(dev)).loss
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); step += 1
                if step % 1000 == 0:
                    print(f"[gpu{gpu}] fold {fk} step {step}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        pv = predict(model, Xtr, b); pt = predict(model, Xte, np.arange(len(te)))
        np.save(os.path.join(OUT, f"gpt2_fold{fk}_va.npy"), pv); np.save(os.path.join(OUT, f"gpt2_fold{fk}_te.npy"), pt)
        print(f"[gpu{gpu}] fold {fk} AUC {roc_auc_score(y[b], pv):.5f} ({time.time()-t0:.0f}s)", flush=True)
        del model, opt; torch.cuda.empty_cache() if dev.type == "cuda" else None


def merge():
    from sklearn.metrics import roc_auc_score
    tr, te, y, folds = load()
    oof = np.zeros(len(tr)); tp = np.zeros(len(te))
    for fk, (a, b) in enumerate(folds):
        oof[b] = np.load(os.path.join(OUT, f"gpt2_fold{fk}_va.npy")); tp += np.load(os.path.join(OUT, f"gpt2_fold{fk}_te.npy")) / 5
    auc = roc_auc_score(y, oof)
    name = "kg_gpt2" if MODEL == "gpt2" else f"kg_{MODEL}"
    np.save(os.path.join(OUT, f"{name}_oof.npy"), oof); np.save(os.path.join(OUT, f"{name}_test.npy"), tp)
    with open(os.path.join(OUT, "log.jsonl"), "a") as f: f.write(json.dumps(dict(name=name, auc=round(auc, 6))) + "\n")
    print(f"RESULT {name} OOF AUC {auc:.6f}", flush=True)


if __name__ == "__main__":
    if mode == "worker":
        g = int(args[args.index("--gpu") + 1]); fl = [int(x) for x in args[args.index("--folds") + 1].split(",")]
        worker(g, fl)
    elif mode == "merge":
        merge()
    else:  # launch: one process per GPU, then merge
        import torch
        ng = max(1, torch.cuda.device_count())
        todo = [fk for fk in range(5) if not os.path.exists(os.path.join(OUT, f"gpt2_fold{fk}_te.npy"))]
        plan = {g: [f for i, f in enumerate(todo) if i % ng == g] for g in range(ng)}
        print("plan:", plan, flush=True)
        procs = [subprocess.Popen([sys.executable, __file__, "worker", "--gpu", str(g), "--folds", ",".join(map(str, fl))]
                                  + ([DATA] if pos else []), env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
                                  if torch.cuda.is_available() else dict(os.environ))
                 for g, fl in plan.items() if fl]
        codes = [p.wait() for p in procs]
        if any(codes): raise SystemExit(f"worker failed: {codes}")
        merge()
