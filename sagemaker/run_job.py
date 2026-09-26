"""Launch, watch and fetch SageMaker training jobs that run the pipeline stages (boto3 only, no notebook).

    python sagemaker/run_job.py --stage smoke --instance ml.m5.large --wait
    python sagemaker/run_job.py --stage preprocess --split train --dicts excl_fold0 --instance ml.m5.4xlarge --wait
    python sagemaker/run_job.py --stage candidates --split train --pre-run <preprocess job name> --fold0-only --instance ml.m5.4xlarge --wait [--spot]
    python sagemaker/run_job.py --fetch <job name> --to artifacts/cloud/<job name>
    python sagemaker/run_job.py --status <job name>      |   --stop <job name>

The job runs sagemaker/entry.py (copied into the source tarball next to src/ and tests/) in the prebuilt
scikit-learn CPU container, which installs requirements.txt first.
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time

import boto3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402,F401  (loads sagemaker/.env if present)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = json.load(open(os.path.join(HERE, "config.json"))) if os.path.exists(os.path.join(HERE, "config.json")) else {}
# prebuilt SageMaker images (CPU: scikit-learn 1.2 / python 3; GPU: PyTorch 2.2 CUDA 12.1); account ids differ per region
CPU_IMAGES = {"us-east-1": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
              "us-east-2": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
              "us-west-2": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
              "eu-west-1": "141502667606.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
              "ap-south-1": "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"}
GPU_IMAGES = {"us-east-1": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
              "us-west-2": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker"}


def package_source(s3, bucket, job):
    """code/business_entity_resolution (src, tests, requirements.txt) plus sagemaker/entry.py -> s3://bucket/code/<job>/sourcedir.tar.gz"""
    buf = io.BytesIO()
    base = os.path.join(ROOT, "code", "business_entity_resolution")
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("src", "tests"):
            for dirpath, dirs, files in os.walk(os.path.join(base, sub)):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in files:
                    if f.endswith(".py"):
                        tar.add(os.path.join(dirpath, f), arcname=os.path.relpath(os.path.join(dirpath, f), base))
        tar.add(os.path.join(base, "requirements.txt"), arcname="requirements.txt")
        tar.add(os.path.join(HERE, "entry.py"), arcname="entry.py")
    buf.seek(0)
    key = f"code/{job}/sourcedir.tar.gz"
    s3.upload_fileobj(buf, bucket, key)
    return f"s3://{bucket}/{key}"


def launch(a):
    session = boto3.Session(region_name=CONFIG["region"])
    sm, s3 = session.client("sagemaker"), session.client("s3")
    bucket = CONFIG["bucket"]
    job = f"ber-{a.stage}-{a.split}-{time.strftime('%Y%m%d-%H%M%S')}"
    source = package_source(s3, bucket, job)
    image = (GPU_IMAGES if a.instance.startswith("ml.g") or a.instance.startswith("ml.p") else CPU_IMAGES)[CONFIG["region"]]
    hp = {"sagemaker_program": "entry.py", "sagemaker_submit_directory": source, "sagemaker_container_log_level": "20", "sagemaker_region": CONFIG["region"],
          "stage": a.stage, "split": a.split, "dicts": a.dicts, "procs": str(a.procs), "bucket": bucket, "job_name": job, "fold0_only": "1" if a.fold0_only else "0",
          "workers": str(a.workers), "buckets": str(a.buckets)}
    channels = [{"ChannelName": "data", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/data/", "S3DataDistributionType": "FullyReplicated"}}},
                {"ChannelName": "artifacts", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/artifacts/", "S3DataDistributionType": "FullyReplicated"}}}]
    if a.pre_run:
        channels.append({"ChannelName": "pre", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/runs/{a.pre_run}/preprocessed/{'fold0' if a.dicts == 'excl_fold0' else a.split}/",
                                                                              "S3DataDistributionType": "FullyReplicated"}}})
    if a.cand_run:
        channels.append({"ChannelName": "pairs", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": f"s3://{bucket}/runs/{a.cand_run}/candidates/{a.split}/",
                                                                                "S3DataDistributionType": "FullyReplicated"}}})
    req = {"TrainingJobName": job, "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"}, "RoleArn": CONFIG["role_arn"],
           "InputDataConfig": channels, "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/runs/{job}/sm-output/"},
           "ResourceConfig": {"InstanceType": a.instance, "InstanceCount": 1, "VolumeSizeInGB": a.volume_gb},
           "StoppingCondition": {"MaxRuntimeInSeconds": a.max_hours * 3600, **({"MaxWaitTimeInSeconds": a.max_hours * 3600 * 2} if a.spot else {})},
           "HyperParameters": {k: json.dumps(v) for k, v in hp.items()}, "EnableManagedSpotTraining": bool(a.spot)}
    sm.create_training_job(**req)
    print(f"launched {job} on {a.instance}{' (spot)' if a.spot else ''}; image {image.split('/')[-1]}")
    if a.wait:
        wait(sm, session, job)
    return job


def wait(sm, session, job, every=60):
    last = None
    while True:
        d = sm.describe_training_job(TrainingJobName=job)
        status, sec = d["TrainingJobStatus"], d.get("SecondaryStatus", "")
        billed = d.get("BillableTimeInSeconds", 0)
        if (status, sec) != last:
            print(f"  {time.strftime('%H:%M:%S')} {status} / {sec} (billable {billed}s)", flush=True)
            last = (status, sec)
        if status in ("Completed", "Failed", "Stopped"):
            if status == "Failed":
                print("  failure reason:", d.get("FailureReason"))
            print_log_tail(session, job)
            return status
        time.sleep(every)


def print_log_tail(session, job, n=40):
    logs = session.client("logs")
    try:
        streams = logs.describe_log_streams(logGroupName="/aws/sagemaker/TrainingJobs", logStreamNamePrefix=job)["logStreams"]
        for st in streams:
            ev = logs.get_log_events(logGroupName="/aws/sagemaker/TrainingJobs", logStreamName=st["logStreamName"], limit=n, startFromHead=False)["events"]
            print(f"--- log {st['logStreamName']} (last {len(ev)} lines)")
            for e in ev:
                print("  ", e["message"][:300])
    except Exception as e:  # noqa: BLE001
        print("  (log tail unavailable:", e, ")")


def fetch(job, to):
    session = boto3.Session(region_name=CONFIG["region"])
    s3 = session.client("s3")
    bucket, prefix = CONFIG["bucket"], f"runs/{job}/"
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if rel.startswith("sm-output/"):
                continue
            path = os.path.join(to, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            s3.download_file(bucket, obj["Key"], path)
            n += 1
    print(f"fetched {n} files to {to}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["smoke", "preprocess", "candidates", "features"])
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--dicts", default="excl_fold0", choices=["excl_fold0", "all_train"])
    p.add_argument("--instance", default="ml.m5.4xlarge")
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--volume-gb", type=int, default=100)
    p.add_argument("--max-hours", type=int, default=6)
    p.add_argument("--spot", action="store_true")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--pre-run", default=None, help="job name whose preprocessed tables feed the candidates stage")
    p.add_argument("--fold0-only", action="store_true", help="candidates for the fold-0 entities only (comparison with the local run)")
    p.add_argument("--cand-run", default=None, help="job name whose candidate_pairs.parquet feeds the features stage")
    p.add_argument("--workers", type=int, default=2, help="features stage: bucket processes in parallel (about 5 GB each)")
    p.add_argument("--buckets", type=int, default=32, help="features stage: number of buckets")
    p.add_argument("--fetch", default=None, metavar="JOB")
    p.add_argument("--to", default=None)
    p.add_argument("--status", default=None, metavar="JOB")
    p.add_argument("--stop", default=None, metavar="JOB")
    a = p.parse_args(argv)
    if a.fetch:
        fetch(a.fetch, a.to or os.path.join(ROOT, "artifacts", "cloud", a.fetch))
    elif a.status:
        d = boto3.Session(region_name=CONFIG["region"]).client("sagemaker").describe_training_job(TrainingJobName=a.status)
        print(d["TrainingJobStatus"], "/", d.get("SecondaryStatus", ""), "billable", d.get("BillableTimeInSeconds", 0), "s")
    elif a.stop:
        boto3.Session(region_name=CONFIG["region"]).client("sagemaker").stop_training_job(TrainingJobName=a.stop)
        print("stop requested")
    elif a.stage:
        launch(a)
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
