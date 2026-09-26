#!/usr/bin/env bash
# Decoy-feature evaluation on fold 0: rebuild the feature matrix with BER_DECOY_FEATURES=1, retrain LightGBM only, score the
# same tuning / report halves as s18 and compare with eda/results/s18_matcher.json (0.9698 on the report half).
# Run from the project root, alone:  bash eda/scripts/run_decoy_eval.sh   (log: eda/results/decoy_eval_log.txt)
set -u
cd "$(dirname "$0")/../.."
export POLARS_MAX_THREADS=8 PYTHONIOENCODING=utf-8
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "1/3 features with the decoy block on -> artifacts/features/fold0_decoy"
BER_DECOY_FEATURES=1 python -u code/business_entity_resolution/src/features.py --pre-dir artifacts/preprocessed/fold0 --prefix train \
  --pairs artifacts/candidates/fold0/final_pairs.parquet --entity-ids eda/results/_fold0_ids.txt --labels artifacts/candidates/fold0/coverage.parquet \
  --out-dir artifacts/features/fold0_decoy --buckets 32 --procs 12 > eda/results/s21_features_log.txt 2>&1 || { log "features FAILED"; tail -5 eda/results/s21_features_log.txt; exit 1; }
tail -1 eda/results/s21_features_log.txt | cut -c1-160

log "2/3 LightGBM on the decoy matrix (same split, same sampling)"
S18_FEAT_DIR=artifacts/features/fold0_decoy S18_MODEL_DIR=artifacts/models/fold0_decoy S18_OUT=eda/results/s21_matcher_decoy.json S18_MODELS=lightgbm \
  python -u eda/scripts/s18_matcher.py > eda/results/s21_matcher_log.txt 2>&1 || { log "matcher FAILED"; tail -5 eda/results/s21_matcher_log.txt; exit 1; }
grep "report half\|best rule\|false merges\|missed matches\|pair-level" eda/results/s21_matcher_log.txt | cut -c1-250

log "3/3 comparison"
python - <<'EOF'
import json
a = json.load(open("eda/results/s18_matcher.json", encoding="utf-8"))["report"]["lightgbm"]
b = json.load(open("eda/results/s21_matcher_decoy.json", encoding="utf-8"))["report"]["lightgbm"]
rows = [("macro F0.5", "macro_f05"), ("singleton empty share", ("singletons", "predicted_empty_share")), ("singleton FPs", ("singletons", "false_positive_entities")),
        ("one-match F0.5", ("one_match", "mean_f05")), ("multi-match F0.5", ("multi_match", "mean_f05")), ("false-merge pairs", "fp"), ("entities with FP", ("false_merges", "entities_with_fp")),
        ("pair recall", "pair_recall"), ("US F0.5", ("by_country", "US", "mean_f05")), ("India F0.5", ("by_country", "India", "mean_f05"))]
def get(d, k):
    if isinstance(k, str): return d[k]
    for x in k: d = d[x]
    return d
print(f"{'metric':24s} {'baseline':>10s} {'decoy':>10s} {'delta':>9s}")
for name, k in rows:
    x, y = get(a, k), get(b, k)
    print(f"{name:24s} {x:>10.4f} {y:>10.4f} {y - x:>+9.4f}" if isinstance(x, float) else f"{name:24s} {x:>10} {y:>10} {y - x:>+9}")
EOF
log "DECOY EVAL DONE"
