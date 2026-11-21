# Running the GPU models on Kaggle (free GPU, 30 h/week)

There are two notebooks. Each saves prediction files on the **same 5 folds** as the local pipeline, so they plug straight into sub02's stack.

| Notebook | Script | What it trains | Est. time on T4 x2 |
|---|---|---|---|
| 1 | `s6e11_part1_trees_nn.py` | XGBoost-from-GLM ×5 encoding seeds, CatBoost-from-GLM ×2, embedding neural net (plain + GLM-residual) ×2 | ~2–3 h |
| 2 | `s6e11_part2_gpt2.py` | GPT-2 reading each row as text (one fold-worker per GPU) | ~2.5–3.5 h |

Both were smoke-tested end to end on a CPU with a tiny sample. The real runs have not been done yet.

## Steps (repeat for each notebook)

1. Open https://www.kaggle.com/competitions/playground-series-s6e11 and choose **Code → New Notebook**. The competition data is attached automatically.
2. In the right panel, under **Session options**:
   - **Accelerator**: GPU T4 x2
   - **Internet**: On. Notebook 2 needs it to download GPT-2, and Kaggle asks for phone verification once.
3. In the first cell, type `%%writefile run.py` on line 1. Below it, paste the **whole** script (part 1 or part 2).
4. In a second cell:
   - Notebook 1: `!python run.py`
   - Notebook 2: `!python run.py launch`
5. Click **Save Version → Save & Run All (Commit)**. It runs in the background (up to 12 h), so you can close the browser.
6. When it finishes, open the version, go to the **Output** tab, and download the `out` folder. You get files like `kg_xgb_glm_t0_oof.npy`, `kg_gpt2_test.npy`, `log.jsonl`.
   - `log.jsonl` lists each model's out-of-fold AUC.

You can run notebook 2 while notebook 1 is still going, if Kaggle lets you. Otherwise run them one after the other.

## Back on your PC (CPU only)

Unzip both downloads into `G:\Kaggle Predicting Credit Card Fraud\kaggle_outputs\`, then:

```powershell
cd "G:\Kaggle Predicting Credit Card Fraud\src"; python import_kaggle.py ..\kaggle_outputs
```

What the import does:
- Checks the files: test id order, lengths, finite values, and AUC > 0.90. Anything broken is skipped.
- Copies the good files into `oof/`.
- Writes three new submission files into `submissions/`:
  - `sub11_sub02_plus_kaggle_C0.1.csv`: sub02's recipe plus the new models. This is the main candidate.
  - `sub12_sub02_plus_kaggle_C0.01.csv`: the same, with stronger regularisation.
  - `sub13_kaggle_plus_glm_bags.csv`: a smaller stack.
- Prints each file's nested-CV AUC next to sub02's 0.95091, so you only submit if CV actually improved.

## Knobs (environment variables, set before `!python`)

- Notebook 2 short on time: `LM_MODEL=distilgpt2` runs about 2× faster and is a bit weaker.
  - Example: `!LM_MODEL=distilgpt2 python run.py launch`
- Notebook 2 interrupted: rerun `!python run.py launch`. Finished folds are skipped (they are saved per fold in `out/`). This only works when you continue in the same session.
- `LM_EPOCHS=2` trains longer: maybe slightly better, but twice the time.
