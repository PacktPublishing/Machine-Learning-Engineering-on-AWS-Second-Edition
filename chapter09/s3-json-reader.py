import boto3
import json
from urllib.parse import urlparse

s3_client = boto3.client("s3")


def load_json_from_s3(s3_path):
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")

    return json.loads(content)


def lambda_handler(event, context):
    results_s3_path = event["s3_path"]

    results = load_json_from_s3(results_s3_path)

    return results
