# Full pipeline, in order. Run from the project root:  powershell -ExecutionPolicy Bypass -File run_all.ps1
# Expect roughly 2-3 hours on a 22-core CPU + RTX 3050 (XGBoost/CatBoost/RealMLP use the GPU).
$ErrorActionPreference = "Stop"
Set-Location src
$P = '{"learning_rate":0.02,"max_depth":3,"min_child_weight":4.235,"subsample":0.745,"colsample_bytree":0.875,"reg_lambda":2.506,"reg_alpha":0.0116,"gamma":0.0123,"max_bin":128}'

python eda.py
python leak_check.py
# raw-feature models
python exp_features.py base eng
python exp_shallow.py
python exp_glm.py 50
python baselines.py lr lr_balanced hgb lgb_balanced rf et xgb cat
# encoded-feature models (in-fold target encoding + digit keys)
python train_te.py glm
python train_te.py glm --pairs
python train_te.py lgb
python train_te.py cat
python train_te.py xgb_glm
python train_te.py xgb_glm --pairs
foreach ($s in 0, 1, 2) {
    python train_te.py lgb_glm --te-seed $s
    python train_te.py xgb_glm _tuned --te-seed $s --params $P
}
python train_te.py lgb_glm --m 80
# neural nets (optional; needs torch + pytabkit)
python train_nn.py --epochs 4
python train_nn.py --allcat --epochs 4
# tuning (results already baked into $P above)
python tune_xgb.py
# ensemble + submission
$members = (Get-Content ..\reports\final_members.txt).Split(" ")
python ensemble.py @members --out submission.csv
python validate_submission.py ..\submission.csv
python ensemble_combos.py
python feature_importance.py
python report.py
