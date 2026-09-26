"""One-time AWS setup: S3 bucket, SageMaker execution role, budget alarm, sagemaker/config.json.

    python sagemaker/setup.py --bucket <unique-name> --region us-east-1 [--budget-usd 100 --email you@example.com]
Requires an AWS CLI profile (aws configure) with SageMaker, S3, IAM and Budgets permissions.  Idempotent.
"""
import argparse
import json
import os
import sys
import time

import boto3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402,F401  (loads sagemaker/.env if present)
from botocore.exceptions import ClientError

ROLE_NAME = "BusinessERSageMakerRole"
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def ensure_bucket(s3, bucket, region):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"bucket {bucket} exists")
    except ClientError:
        kwargs = {} if region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": region}}
        s3.create_bucket(Bucket=bucket, **kwargs)
        s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={"BlockPublicAcls": True, "IgnorePublicAcls": True,
                                                                                  "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        print(f"bucket {bucket} created (private)")


def ensure_role(iam, bucket):
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
        print(f"role {ROLE_NAME} exists")
    except ClientError:
        role = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust), Description="SageMaker execution role for the entity-resolution pipeline")["Role"]
        print(f"role {ROLE_NAME} created")
        time.sleep(10)
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess")
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"], "Resource": f"arn:aws:s3:::{bucket}"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], "Resource": f"arn:aws:s3:::{bucket}/*"},
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"], "Resource": "*"},
    ]}
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="BusinessERBucketAccess", PolicyDocument=json.dumps(policy))
    return role["Arn"]


def ensure_budget(session, account, usd, email):
    if not email:
        print("no e-mail given: budget alarm skipped")
        return
    budgets = session.client("budgets", region_name="us-east-1")
    name = "business-er-credits"
    try:
        budgets.describe_budget(AccountId=account, BudgetName=name)
        print(f"budget {name} exists")
        return
    except ClientError:
        pass
    budgets.create_budget(AccountId=account,
                          Budget={"BudgetName": name, "BudgetLimit": {"Amount": str(usd), "Unit": "USD"}, "TimeUnit": "MONTHLY", "BudgetType": "COST"},
                          NotificationsWithSubscribers=[{"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": t, "ThresholdType": "PERCENTAGE"},
                                                         "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}]} for t in (50, 80)])
    print(f"budget {name}: alerts at 50% and 80% of ${usd} to {email}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--budget-usd", type=float, default=100)
    p.add_argument("--email", default=None)
    a = p.parse_args(argv)
    session = boto3.Session(region_name=a.region)
    account = session.client("sts").get_caller_identity()["Account"]
    print(f"account {account}, region {a.region}")
    ensure_bucket(session.client("s3"), a.bucket, a.region)
    role_arn = ensure_role(session.client("iam"), a.bucket)
    ensure_budget(session, account, a.budget_usd, a.email)
    json.dump({"bucket": a.bucket, "region": a.region, "role_arn": role_arn, "account": account}, open(CONFIG, "w"), indent=1)
    print(f"wrote {CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
