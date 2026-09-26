#!/usr/bin/env bash
# Test-time pipeline, end to end: preprocess (all-train dictionaries) -> candidates -> features + LightGBM scoring -> matching_results.tsv -> validator.
# Run from the project root:  bash eda/scripts/run_test_pipeline.sh    (log: eda/results/test_pipeline_log.txt)
set -u
cd "$(dirname "$0")/../.."
export POLARS_MAX_THREADS=8 PYTHONIOENCODING=utf-8
SRC=code/business_entity_resolution/src
log() { echo "[$(date +%H:%M:%S)] $*"; }
STAGE="${1:-preprocess}"

if [ "$STAGE" = "preprocess" ]; then
  log "1/5 preprocess test with the all-train dictionaries"
  mkdir -p artifacts/preprocessed/test
  python -u $SRC/preprocess.py --split-dir student_resource/dataset/test --prefix test --out-dir artifacts/preprocessed/test \
    --translit artifacts/translit_dict_all_train.json --vocab artifacts/address_vocab_all_train.json --procs 12 --chunk 40000 \
    > artifacts/preprocessed/test/preprocess_log.txt 2>&1 || { log "preprocess FAILED"; tail -5 artifacts/preprocessed/test/preprocess_log.txt; exit 1; }
  tail -2 artifacts/preprocessed/test/preprocess_log.txt
  STAGE=candidates
fi

if [ "$STAGE" = "candidates" ]; then
  log "2/5 candidate generation (13 strategies)"
  mkdir -p output
  python -u $SRC/generate_candidates.py --pre-dir artifacts/preprocessed/test --prefix test --out output/candidate_pairs.tsv --procs 12 \
    > eda/results/test_candidates_log.txt 2>&1 || { log "candidates FAILED"; tail -5 eda/results/test_candidates_log.txt; exit 1; }
  grep "union\|wrote" eda/results/test_candidates_log.txt | cut -c1-200
  STAGE=score
fi

if [ "$STAGE" = "score" ]; then
  log "3/5 features + LightGBM scoring per bucket"
  mkdir -p artifacts/scores/test
  python -u $SRC/features.py --pre-dir artifacts/preprocessed/test --prefix test --pairs output/candidate_pairs.parquet --out-dir artifacts/scores/test \
    --buckets 128 --procs 12 --model artifacts/models/fold0/lightgbm.txt --feature-list artifacts/features/fold0/features_meta.json --neg-rate 0.065 \
    > eda/results/test_scoring_log.txt 2>&1 || { log "scoring FAILED"; tail -5 eda/results/test_scoring_log.txt; exit 1; }
  tail -2 eda/results/test_scoring_log.txt | cut -c1-200
  STAGE=submission
fi

if [ "$STAGE" = "submission" ]; then
  log "4/5 matching_results.tsv (threshold 0.75 + pool uniqueness)"
  python -u $SRC/submission.py --pred-dir artifacts/scores/test --pre-dir artifacts/preprocessed/test --prefix test --out output/matching_results.tsv \
    --threshold 0.75 --stats eda/results/test_submission_stats.json || { log "submission FAILED"; exit 1; }
  log "5/5 organisers' validator"
  (cd student_resource && python utils/validate_submission.py --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test) \
    > eda/results/test_validator_log.txt 2>&1; echo "validator exit $?" >> eda/results/test_validator_log.txt
  tail -5 eda/results/test_validator_log.txt
fi
log "PIPELINE DONE"
