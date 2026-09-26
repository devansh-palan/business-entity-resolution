"""Upload the competition data and the learned dictionaries to S3 (skips files already present with the same size).

    python sagemaker/upload_data.py [--include-preprocessed]
"""
import argparse
import json
import os
import sys

import boto3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402,F401  (loads sagemaker/.env if present)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = json.load(open(os.path.join(HERE, "config.json")))


def upload_tree(s3, local_dir, prefix, patterns=None):
    bucket = CONFIG["bucket"]
    n = 0
    for dirpath, _, files in os.walk(local_dir):
        for f in files:
            if patterns and not any(f.endswith(x) for x in patterns):
                continue
            path = os.path.join(dirpath, f)
            key = prefix + "/" + os.path.relpath(path, local_dir).replace("\\", "/")
            size = os.path.getsize(path)
            try:
                if s3.head_object(Bucket=bucket, Key=key)["ContentLength"] == size:
                    print(f"  = {key} ({size / 1e6:.0f} MB, already there)")
                    continue
            except s3.exceptions.ClientError:
                pass
            print(f"  ^ {key} ({size / 1e6:.0f} MB)", flush=True)
            s3.upload_file(path, bucket, key)
            n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--include-preprocessed", action="store_true", help="also upload artifacts/preprocessed (2.9 GB per split)")
    a = p.parse_args(argv)
    s3 = boto3.Session(region_name=CONFIG["region"]).client("s3")
    print(f"bucket s3://{CONFIG['bucket']}")
    upload_tree(s3, os.path.join(ROOT, "student_resource", "dataset"), "data", [".tsv"])
    upload_tree(s3, os.path.join(ROOT, "artifacts", "validation"), "artifacts/validation")
    for f in os.listdir(os.path.join(ROOT, "artifacts")):
        if f.endswith(".json"):
            path = os.path.join(ROOT, "artifacts", f)
            s3.upload_file(path, CONFIG["bucket"], f"artifacts/{f}")
            print(f"  ^ artifacts/{f}")
    if a.include_preprocessed:
        upload_tree(s3, os.path.join(ROOT, "artifacts", "preprocessed"), "artifacts/preprocessed", [".parquet"])
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
