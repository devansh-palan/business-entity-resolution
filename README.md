# Business entity resolution (Amazon ML Challenge 2026)

Match every Source 1 business entity to its records in Source 2 and Source 3, across countries, with noisy
names and addresses. The score is the macro F0.5 over Source 1 entities (a singleton scores 1.0 when its
predicted set is empty). This repository holds the full pipeline, every analysis report with the numbers behind
each decision, the trained fold-0 models, and the small artifacts needed to reproduce the current submission
(expected macro F0.5 around 0.97 on held-out training data).

Competition rules that shape the code: no external lookups (no geocoding, no entity APIs, no internet
augmentation); the final model must be MIT or Apache licensed and at most 8B parameters; `candidate_pairs.tsv`
is the FINAL candidate set fed to the model; the country field is an open-set string and the test set contains
France; the test set is never used for modelling or threshold selection; pair-level metrics are never optimised
directly.

## Current result

| Stage | Macro F0.5 (fold 0, report half) | Notes |
|---|---|---|
| Deterministic rule baseline | 0.612 | `src/rules.py`, `eda/rule_baseline.md` |
| Logistic regression | 0.9191 | 1.5M sampled pairs |
| XGBoost | 0.9696 | 1,509 trees, depth 8 |
| **LightGBM (submitted)** | **0.9701** | 1,056 trees, 127 leaves, rule: p >= 0.75 + pool uniqueness |
| Oracle that knows each entity's true match count | +0.0038 | upper bound for a better decision layer |

Candidate recall on fold 0 is 0.9895 with 91 candidates per entity (13 strategies). The submitted
`matching_results.tsv` has 1,732,544 rows, 5,724,338 matched ids, 5.63% empty rows (train has 5.6% singletons).

Where the remaining error is (report half of fold 0, 55,170 entities), from `eda/matcher_baseline.md`:

- 1,851 false merges: 49% share the exact normalised name with the entity, 58% have a house-number conflict.
  788 are records that belong to another Source 1 entity, 1,063 belong to no entity ("ownerless decoys").
- 9,504 true pairs are candidates but score below 0.75.
- 2,095 true pairs are missing from the candidate set (`eda/candidate_misses.md` lists the causes and the fixes
  that were tried and rejected).

Things already tried at the decision layer that did NOT help (see `eda/threshold_sweep.md` and
`eda/decision_rules.md` before repeating them): per-segment thresholds, top-1 only, gap cuts, per-entity
F1-style cuts, an automatic name+house+street rule, a separate singleton model. A single global threshold
(plateau 0.63 to 0.80) plus one-to-one pool uniqueness is the best validated rule.

## Layout

```
code/business_entity_resolution/
  src/            the pipeline (see the stage table below)
  tests/          99 pytest tests (2 skipped); run them after any change
  requirements.txt
eda/              one markdown report per stage, in the order the work was done
  scripts/        the scripts that produced the reports (s0 .. s20) and the two runner scripts
  results/        the JSON numbers and logs every report quotes
artifacts/
  models/fold0/   lightgbm.txt (submitted), xgboost.json, logistic_regression.joblib, models_info.json
  features/fold0/features_meta.json      the 157 feature names in model order (needed for scoring)
  validation/     fold assignments (entity id lists), summary, dictionaries learned without fold 0
  candidates/fold0/   coverage.parquet (fold-0 true pairs as candidate positions), miss tables
  *_all_train.json    transliteration dictionary and address vocabulary learned on all of train (test time)
sagemaker/        AWS kit (SageMaker and EC2 job launchers); needs your own account, see sagemaker/README.md
student_resource/ organisers' README, documentation template and validator; put the dataset here (not committed)
```

Pipeline stages, each a plain Python CLI in `src/`:

| Step | Module | Produces |
|---|---|---|
| 0 | `validation.py build` | entity-level 5-fold split with address-merge grouping, positive pairs |
| 1 | `preprocess.py` (`names.py`, `addresses.py`, `translit.py`, `normalize.py`) | `<prefix>_source{1,2,3}.parquet` with 32 normalised columns |
| 2 | `generate_candidates.py` (`candidates.py`) | `candidate_pairs.tsv` + `candidate_pairs.parquet` (q_idx, p_idx) |
| 3 | `features.py` | 157 label-free pair features in 32 buckets, or LightGBM scores with `--model` |
| 4 | `eda/scripts/s18_matcher.py` (`matcher.py`) | the three models, validation scores, `eda/results/s18_matcher.json` |
| 5 | `submission.py` | `matching_results.tsv` from the scores (threshold + pool uniqueness) |
| eval | `evaluation.py` | the official macro F0.5, entity-level breakdowns |

`features.py` groups: name (51), address (50), TF-IDF cosines (4), context (4), cross (13), entity-relative
rank/gap features (18), missingness (16), source (1). Every pool statistic is fitted unsupervised on the pool
records, so nothing leaks the labels. A decoy-feature block exists behind `BER_DECOY_FEATURES=1` (house-number
and unit conflicts, exact-twin counts); it is unevaluated, `eda/scripts/run_decoy_eval.sh` is the planned test.

## Setup

Python 3.12. The dataset is not in this repository: copy the organisers' `student_resource/dataset/` folder
(train and test TSVs, 2.2 GB) into `student_resource/`.

```
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution && python -m pytest -q && cd ../..
```

All commands below run from the repository root.

## Reproducing fold-0 validation (train side)

```
# 0. folds and positive pairs (entity_folds.parquet, positive_pairs.parquet, fold*_ground_truth.tsv)
python code/business_entity_resolution/src/validation.py build --train-dir student_resource/dataset/train --out-dir artifacts/validation

# 1. preprocess train with the dictionaries learned WITHOUT fold 0 (about 5 min)
python code/business_entity_resolution/src/preprocess.py --split-dir student_resource/dataset/train --prefix train \
  --out-dir artifacts/preprocessed/fold0 --translit artifacts/validation/translit_dict_excl_fold0.json \
  --vocab artifacts/validation/address_vocab_excl_fold0.json --procs 12 --chunk 40000

# 2. candidates for the fold-0 entities (13 strategies; recall 0.9895, 40.2M pairs)
python code/business_entity_resolution/src/generate_candidates.py --pre-dir artifacts/preprocessed/fold0 --prefix train \
  --out artifacts/candidates/fold0/candidate_pairs.tsv --entity-ids artifacts/validation/fold0_entity_ids.txt --procs 12

# 3. the feature matrix with labels (26 min, 3.3 GB, 5 GB RAM peak per bucket)
python code/business_entity_resolution/src/features.py --pre-dir artifacts/preprocessed/fold0 --prefix train \
  --pairs artifacts/candidates/fold0/candidate_pairs.parquet --entity-ids artifacts/validation/fold0_entity_ids.txt \
  --labels artifacts/candidates/fold0/coverage.parquet --out-dir artifacts/features/fold0 --buckets 32 --procs 12

# 4. models, entity-level evaluation, threshold sweep, decision rules
S18_MODELS=lightgbm python eda/scripts/s18_matcher.py     # one model family at a time on an 8 GB machine
python eda/scripts/s19_threshold_sweep.py
python eda/scripts/s20_decision_rules.py
```

The split inside fold 0 (441,365 entities) is by entity: 24 feature buckets train the models, bucket group A
(55,171 entities) picks the threshold, bucket group B (55,170 entities, 3,150 singletons) reports the number.
Never pick a threshold on the report half.

## Reproducing the submission (test side)

```
bash eda/scripts/run_test_pipeline.sh                # all stages, about 4 h 40 min on an 8 GB laptop
bash eda/scripts/run_test_pipeline.sh score          # restart from a stage: preprocess | candidates | score | submission
```

Stages: preprocess test with the all-train dictionaries (7 min), candidates (161.9M pairs, 93 per entity),
features + LightGBM scoring in 128 buckets (about 71 s each), `submission.py` with threshold 0.75, then the
organisers' validator. The validator loads every id into Python sets and needs more than 9 GB of RAM; on smaller
machines check the files with the streaming rules in `src/submission.py` instead.

## Artifacts not in this repository

| Artifact | Size | How to get it |
|---|---|---|
| `artifacts/validation/entity_folds.parquet`, `positive_pairs.parquet`, `fold*_ground_truth.tsv` | 250 MB | step 0 above (deterministic, seed 0) |
| `artifacts/preprocessed/fold0/` and `/test/` | 2.8 GB each | step 1 (5 to 7 min) |
| `artifacts/candidates/fold0/candidate_pairs.parquet` (40.2M pairs) | 138 MB | step 2 |
| `artifacts/features/fold0/part_00..31.parquet` + `_pool_stats/` | 3.3 GB | step 3 (26 min) |
| `artifacts/models/fold0/val_scores.parquet` (scores of all three models on fold 0) | 145 MB | step 4 |
| `artifacts/scores/test/pred_*.parquet`, `output/candidate_pairs.tsv`, `output/matching_results.tsv` | 1.6 GB + 2.2 GB | the test run |

The large files are also available from the repository owner on request if you want to skip the compute.

## Working on an 8 GB machine

The pipeline was built on a Windows laptop with about 8 GB of free RAM, which is why every heavy step runs in
a fresh child process (`ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1)`): Polars' allocator keeps
freed memory, so a long-lived parent would grow until it dies. Run one heavy job at a time; even the test
suite started during a feature bucket's peak has killed the bucket. With more RAM, raise `--workers` in
`features.py` (about 4 GB per concurrent bucket) and fit all three models in one `s18_matcher.py` run.

## Reading the analysis

The reports in `eda/` are in working order and each one ends with the decision it justified:
`EDA_report.md`, `problem_structure.md`, `validation_design.md`, `name_normalization.md`,
`address_analysis.md`, `rule_baseline.md`, `candidate_generation.md`, `candidate_misses.md`,
`pair_features.md`, `matcher_baseline.md`, `threshold_sweep.md`, `decision_rules.md`.

The early scripts (`s0` to `s13`) read a parquet cache of the raw TSVs under `BER_SCRATCH`
(default `artifacts/_scratch`), which `s0_inventory.py` builds; later scripts read `artifacts/` directly.

## Secrets

`sagemaker/.env`, `sagemaker/remote.env` and `sagemaker/config.json` are git-ignored and must stay that way;
copy the `.example` files and fill in your own account. Nothing in this repository contains credentials.
