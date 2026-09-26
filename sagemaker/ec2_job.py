"""Run a pipeline stage on a plain EC2 instance (fallback while the SageMaker training quotas are pending).

    python sagemaker/ec2_job.py --stage smoke --instance r5.2xlarge --wait
    python sagemaker/ec2_job.py --stage preprocess --split train --dicts excl_fold0 --instance r5.2xlarge --wait
    python sagemaker/ec2_job.py --stage candidates --split train --pre-run <job> --fold0-only --instance r5.2xlarge --wait
    python sagemaker/ec2_job.py --status <job>   |   --terminate <job>

The instance boots Amazon Linux 2023, installs Python 3.12 and requirements.txt, downloads the code tarball and the
S3 data, runs sagemaker/entry.py with BER_* environment variables, uploads outputs to s3://<bucket>/runs/<job>/,
and terminates itself.  Progress is written to s3://<bucket>/runs/<job>/log.txt every minute, which --wait tails.
An instance profile with access to the bucket is created on first use.  The default on-demand quota of this
account is 8 vCPUs, so r5.2xlarge (8 vCPU, 64 GB, ~$0.50/h) is the largest CPU instance available today.
"""
import argparse
import base64
import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import _env  # noqa: E402,F401
from run_job import CONFIG, package_source  # noqa: E402

PROFILE = "BusinessEREC2Profile"
ROLE = "BusinessEREC2Role"

USER_DATA = r"""#!/bin/bash
set -euo pipefail
exec > /var/log/ber.log 2>&1
BUCKET={bucket}; JOB={job}
log() {{ echo "[$(date +%H:%M:%S)] $*"; aws s3 cp /var/log/ber.log s3://$BUCKET/runs/$JOB/log.txt --quiet || true; }}
( while true; do sleep 60; aws s3 cp /var/log/ber.log s3://$BUCKET/runs/$JOB/log.txt --quiet || true; done ) &
log "instance up: $(nproc) vcpu, $(free -g | awk '/Mem/{{print $2}}') GB"
dnf install -y -q python3.12 python3.12-pip python3.12-devel gcc gcc-c++ > /dev/null
mkdir -p /work/code /work/data /work/artifacts /work/pre /work/pairs
aws s3 cp s3://$BUCKET/code/$JOB/sourcedir.tar.gz /work/sourcedir.tar.gz --quiet
tar -xzf /work/sourcedir.tar.gz -C /work/code
python3.12 -m pip install -q -r /work/code/requirements.txt
log "dependencies installed"
aws s3 sync s3://$BUCKET/data/ /work/data/ --quiet
aws s3 sync s3://$BUCKET/artifacts/ /work/artifacts/ --quiet
{pre_sync}
{cand_sync}
log "inputs synced"
export BER_STAGE={stage} BER_SPLIT={split} BER_DICTS={dicts} BER_PROCS={procs} BER_BUCKET=$BUCKET BER_JOB_NAME=$JOB BER_FOLD0_ONLY={fold0} BER_WORKERS={workers} BER_BUCKETS={buckets}
export BER_DATA_DIR=/work/data BER_ARTS_DIR=/work/artifacts BER_PRE_DIR=/work/pre BER_PAIRS_DIR=/work/pairs BER_WORK_DIR=/work/out
cd /work/code && python3.12 entry.py || log "STAGE FAILED"
log "stage finished"
aws s3 cp /var/log/ber.log s3://$BUCKET/runs/$JOB/log.txt --quiet || true
shutdown -h now
"""


def ensure_profile(session, bucket):
    iam = session.client("iam")
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        iam.get_role(RoleName=ROLE)
    except ClientError:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust))
        time.sleep(8)
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"], "Resource": f"arn:aws:s3:::{bucket}/*"}]}
    iam.put_role_policy(RoleName=ROLE, PolicyName="BusinessERBucket", PolicyDocument=json.dumps(policy))
    try:
        iam.get_instance_profile(InstanceProfileName=PROFILE)
    except ClientError:
        iam.create_instance_profile(InstanceProfileName=PROFILE)
        iam.add_role_to_instance_profile(InstanceProfileName=PROFILE, RoleName=ROLE)
        time.sleep(12)
    return PROFILE


def latest_al2023_ami(session):
    ssm = session.client("ssm")
    return ssm.get_parameter(Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64")["Parameter"]["Value"]


def launch(a):
    session = boto3.Session(region_name=CONFIG["region"])
    ec2, s3 = session.client("ec2"), session.client("s3")
    bucket = CONFIG["bucket"]
    job = f"ec2-{a.stage}-{a.split}-{time.strftime('%Y%m%d-%H%M%S')}"
    package_source(s3, bucket, job)
    profile = ensure_profile(session, bucket)
    pre_sync = f"aws s3 sync s3://$BUCKET/runs/{a.pre_run}/preprocessed/{'fold0' if a.dicts == 'excl_fold0' else a.split}/ /work/pre/ --quiet" if a.pre_run else "true"
    cand_sync = f"aws s3 sync s3://$BUCKET/runs/{a.cand_run}/candidates/{a.split}/ /work/pairs/ --quiet" if a.cand_run else "true"
    user_data = USER_DATA.format(bucket=bucket, job=job, stage=a.stage, split=a.split, dicts=a.dicts, procs=a.procs, fold0="1" if a.fold0_only else "0", pre_sync=pre_sync,
                                 cand_sync=cand_sync, workers=a.workers, buckets=a.buckets)
    r = ec2.run_instances(ImageId=latest_al2023_ami(session), InstanceType=a.instance, MinCount=1, MaxCount=1,
                          IamInstanceProfile={"Name": profile}, UserData=user_data, InstanceInitiatedShutdownBehavior="terminate",
                          BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": a.volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                          TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": job}, {"Key": "project", "Value": "business-er"}]}],
                          **({"InstanceMarketOptions": {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}}} if a.spot else {}))
    iid = r["Instances"][0]["InstanceId"]
    print(f"launched {job}: instance {iid} ({a.instance}{', spot' if a.spot else ''}); outputs -> s3://{bucket}/runs/{job}/")
    json.dump({"job": job, "instance_id": iid}, open(os.path.join(HERE, f".{job}.json"), "w"))
    if a.wait:
        wait(session, job, iid)
    return job


def wait(session, job, iid, every=60):
    ec2, s3 = session.client("ec2"), session.client("s3")
    seen = 0
    while True:
        try:
            state = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
        except ClientError:
            state = "unknown"
        try:
            body = s3.get_object(Bucket=CONFIG["bucket"], Key=f"runs/{job}/log.txt")["Body"].read().decode("utf-8", "replace")
            lines = body.splitlines()
            for line in lines[seen:]:
                print("  ", line[:200], flush=True)
            seen = len(lines)
        except ClientError:
            pass
        if state in ("terminated", "shutting-down"):
            print(f"instance {state}")
            return
        time.sleep(every)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["smoke", "preprocess", "candidates", "features"])
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--dicts", default="excl_fold0", choices=["excl_fold0", "all_train"])
    p.add_argument("--instance", default="r5.2xlarge")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--volume-gb", type=int, default=120)
    p.add_argument("--spot", action="store_true")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--pre-run", default=None)
    p.add_argument("--fold0-only", action="store_true")
    p.add_argument("--cand-run", default=None, help="job name whose candidate_pairs.parquet feeds the features stage")
    p.add_argument("--workers", type=int, default=4, help="features stage: bucket processes in parallel (about 5 GB each; 64 GB box -> 4)")
    p.add_argument("--buckets", type=int, default=32)
    p.add_argument("--status", default=None, metavar="JOB")
    p.add_argument("--terminate", default=None, metavar="JOB")
    a = p.parse_args(argv)
    session = boto3.Session(region_name=CONFIG["region"])
    if a.status or a.terminate:
        job = a.status or a.terminate
        meta = json.load(open(os.path.join(HERE, f".{job}.json")))
        if a.terminate:
            session.client("ec2").terminate_instances(InstanceIds=[meta["instance_id"]])
            print("terminate requested")
        else:
            print(session.client("ec2").describe_instances(InstanceIds=[meta["instance_id"]])["Reservations"][0]["Instances"][0]["State"]["Name"])
    elif a.stage:
        launch(a)
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
