"""Entry point executed inside the SageMaker training container (script mode).

SageMaker mounts the input channels under /opt/ml/input/data/<channel>/ and passes hyperparameters as JSON; this
script runs one pipeline stage and uploads its outputs directly to s3://<bucket>/runs/<job-name>/ (large outputs
are not routed through /opt/ml/output, whose tarball would be slow for gigabytes of parquet).

Stages:
  smoke        print the environment, run the unit tests, write a marker (validates the container for a few cents)
  preprocess   preprocess a split (train or test) with the given dictionaries -> preprocessed parquet tables
  candidates   generate candidate_pairs.tsv for a split from preprocessed tables (from S3 or from this job's preprocess)
"""
import json
import os
import subprocess
import sys
import time

import boto3

T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
HP = json.load(open("/opt/ml/input/config/hyperparameters.json")) if os.path.exists("/opt/ml/input/config/hyperparameters.json") else {}
HP = {k: v.strip('"') for k, v in HP.items()}
HP = {**{k[4:].lower(): v for k, v in os.environ.items() if k.startswith("BER_")}, **HP}   # EC2 runs pass the same settings as BER_* variables
STAGE = HP.get("stage", "smoke")
BUCKET, JOB = HP.get("bucket", ""), HP.get("job_name", "job")
SPLIT, DICTS = HP.get("split", "train"), HP.get("dicts", "excl_fold0")
PROCS = int(HP.get("procs", "8"))
DATA = HP.get("data_dir", "/opt/ml/input/data/data")          # the TSVs (train/ and test/ sub-folders)
ARTS = HP.get("arts_dir", "/opt/ml/input/data/artifacts")     # dictionaries and folds
PRE_IN = HP.get("pre_dir", "/opt/ml/input/data/pre")          # optional: preprocessed tables of a previous job
PAIRS_IN = HP.get("pairs_dir", "/opt/ml/input/data/pairs")    # optional: candidate_pairs.parquet (and labels.parquet) of a previous candidates job
WORK = HP.get("work_dir", "/opt/ml/work")
os.makedirs(WORK, exist_ok=True)
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, "src"))
s3 = boto3.client("s3")


def upload_dir(local_dir, prefix):
    n = 0
    for dirpath, _, files in os.walk(local_dir):
        for f in files:
            path = os.path.join(dirpath, f)
            key = f"runs/{JOB}/{prefix}/{os.path.relpath(path, local_dir)}".replace("\\", "/")
            s3.upload_file(path, BUCKET, key)
            n += 1
    log(f"uploaded {n} files to s3://{BUCKET}/runs/{JOB}/{prefix}/")


def stage_smoke():
    import multiprocessing
    import polars, numpy, scipy
    log(f"python {sys.version.split()[0]}, polars {polars.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}, cpus {multiprocessing.cpu_count()}")
    try:
        import psutil
        log(f"RAM total {psutil.virtual_memory().total / 1e9:.1f} GB")
    except Exception:
        pass
    r = subprocess.run([sys.executable, "-m", "pytest", os.path.join(CODE, "tests"), "-q"], capture_output=True, text=True)
    log("pytest tail:", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-500:])
    for ch in (DATA, ARTS):
        if os.path.isdir(ch):
            for dirpath, _, files in os.walk(ch):
                for f in files:
                    log(f"  input {os.path.relpath(os.path.join(dirpath, f), '/opt/ml/input/data')} {os.path.getsize(os.path.join(dirpath, f)) / 1e6:.0f} MB")
    open(os.path.join(WORK, "smoke.json"), "w").write(json.dumps({"ok": r.returncode == 0, "stage": "smoke"}))
    upload_dir(WORK, "smoke")


def stage_preprocess():
    import preprocess as pp
    translit = os.path.join(ARTS, "validation", "translit_dict_excl_fold0.json") if DICTS == "excl_fold0" else os.path.join(ARTS, "translit_dict_all_train.json")
    vocab = os.path.join(ARTS, "validation", "address_vocab_excl_fold0.json") if DICTS == "excl_fold0" else os.path.join(ARTS, "address_vocab_all_train.json")
    out_dir = os.path.join(WORK, "preprocessed", SPLIT if DICTS != "excl_fold0" else "fold0")
    pp.preprocess_split(os.path.join(DATA, SPLIT), SPLIT, out_dir, translit, vocab, n_procs=PROCS, log=log)
    upload_dir(out_dir, f"preprocessed/{os.path.basename(out_dir)}")


def stage_candidates():
    import generate_candidates as gc_
    pre_dir = PRE_IN if os.path.isdir(PRE_IN) else os.path.join(WORK, "preprocessed", SPLIT if DICTS != "excl_fold0" else "fold0")
    if not os.path.isdir(pre_dir):
        stage_preprocess()
    out = os.path.join(WORK, "candidates", SPLIT, "candidate_pairs.tsv")
    ids = None
    ids_file = os.path.join(ARTS, "validation", "fold0_entity_ids.txt")
    if HP.get("fold0_only", "0") == "1" and os.path.exists(ids_file):
        ids = [line.strip() for line in open(ids_file, encoding="utf-8") if line.strip()]
    stats = gc_.generate(pre_dir, SPLIT, out, ids, n_threads=PROCS, log=log)
    json.dump(stats, open(os.path.join(WORK, "candidates", SPLIT, "stats.json"), "w"), indent=1)
    upload_dir(os.path.join(WORK, "candidates"), "candidates")


def stage_features():
    """Pairwise feature matrix for the candidate pairs of a split (from a previous candidates job, or generated here)."""
    import features as ft
    pre_dir = PRE_IN if os.path.isdir(PRE_IN) else os.path.join(WORK, "preprocessed", SPLIT if DICTS != "excl_fold0" else "fold0")
    pairs = os.path.join(PAIRS_IN, "candidate_pairs.parquet") if os.path.isdir(PAIRS_IN) else os.path.join(WORK, "candidates", SPLIT, "candidate_pairs.parquet")
    if not os.path.exists(pairs):
        stage_candidates()
    labels = os.path.join(PAIRS_IN, "labels.parquet") if os.path.exists(os.path.join(PAIRS_IN, "labels.parquet")) else None
    ids = None
    ids_file = os.path.join(ARTS, "validation", "fold0_entity_ids.txt")
    if HP.get("fold0_only", "0") == "1" and os.path.exists(ids_file):
        ids = [line.strip() for line in open(ids_file, encoding="utf-8") if line.strip()]
    out = os.path.join(WORK, "features", SPLIT)
    ft.build(pre_dir, SPLIT, pairs, out, ids, labels, n_buckets=int(HP.get("buckets", "32")), workers=-1, log=log, n_procs=int(HP.get("workers", "2")))
    upload_dir(out, f"features/{SPLIT}")


if __name__ == "__main__":
    log(f"stage {STAGE}, split {SPLIT}, dicts {DICTS}, job {JOB}, bucket {BUCKET}")
    {"smoke": stage_smoke, "preprocess": stage_preprocess, "candidates": stage_candidates, "features": stage_features}[STAGE]()
    log("done")
