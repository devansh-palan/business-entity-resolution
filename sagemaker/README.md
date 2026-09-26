# Running the pipeline on SageMaker

Everything here is driven from the terminal with boto3; no notebook is needed. The pipeline code is plain
Python (Polars, numpy, scipy, sparse_dot_topn, rapidfuzz) and runs unchanged inside SageMaker's prebuilt
containers.

## One-time setup (your side)

1. Install the AWS CLI v2 and run `aws configure` with an access key that can use SageMaker, S3 and IAM
   (attach `AmazonSageMakerFullAccess` and `AmazonS3FullAccess` to the IAM user, plus `IAMFullAccess` once so the
   execution role can be created; it can be removed afterwards). Region: the one your credits apply to
   (`us-east-1` unless told otherwise). Output format: json.
2. Enable Cost Explorer once in the console (Billing > Cost Explorer) if you want spend reports; the budget
   alarm below works without it.

## One-time setup (from this terminal, after step 1)

```
python sagemaker/setup.py --bucket <unique-bucket-name> --region us-east-1
```
creates the bucket, the SageMaker execution role (`BusinessERSageMakerRole`), a monthly budget alarm at $50 and
$80 e-mailed to the address you give, and writes `sagemaker/config.json`.

```
python sagemaker/upload_data.py
```
uploads `student_resource/dataset/{train,test}/*.tsv` (2.2 GB) and `artifacts/{validation,*.json}` (the
learned dictionaries and folds, 300 MB) to `s3://<bucket>/data/` and `s3://<bucket>/artifacts/`.

## Running a stage

```
python sagemaker/run_job.py --stage smoke --instance ml.m5.large --wait          # 2 minutes, a few cents
python sagemaker/run_job.py --stage preprocess --split train --dicts excl_fold0 --instance ml.m5.4xlarge --wait
python sagemaker/run_job.py --stage candidates --split train --pre-run <job-name-of-preprocess> --instance ml.m5.4xlarge --wait
```
Each job packages `code/business_entity_resolution` into a source tarball, installs `requirements.txt` in the
container, runs `sagemaker/entry.py` for the stage, and copies its outputs to `s3://<bucket>/runs/<job-name>/`.
`--spot` requests managed spot capacity (60-70% cheaper, may wait for capacity); `--wait` polls until the job
ends and prints the tail of its log. Jobs stop themselves at the end, so nothing idles.

Instances used: `ml.m5.4xlarge` (16 vCPU, 64 GB, $0.92/h) for the CPU stages; `ml.g5.2xlarge` (A10G 24 GB, $1.52/h)
only for the reranker stage once it exists.

## Fetching results

```
python sagemaker/run_job.py --fetch <job-name> --to artifacts/cloud/<job-name>
```
downloads the run's outputs (logs, JSON results, candidate files) to this machine.
