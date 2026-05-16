import boto3
import random
import string
from time import sleep
from contextlib import contextmanager

sm = boto3.client("sagemaker")


@contextmanager
def block(label):
    print(f"[{label}]: START")
    yield
    print(f"[{label}]: END")


def generate_string(length=6):
    return ''.join(
        random.choices(
            string.ascii_lowercase,
            k=length
        )
    )


def wait_for_endpoint(endpoint_name, sagemaker_client=sm):
    while True:
        response = sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        status = response["EndpointStatus"]
        print(f"ENDPOINT STATUS: {status}")

        if status == "InService":
            break
        elif status == "Failed":
            raise RuntimeError("ENDPOINT CREATION FAILED")

        sleep(15)


def lambda_handler(event, context):
    endpoint_name = event["endpoint_name"]
    model_arn = event["model_arn"]
    instance_type = event.get("instance_type", "ml.g5.4xlarge")

    model_name = model_arn.split("/")[-1]
    unique = generate_string()
    endpoint_config_name = f"endpoint-config-{unique}"

    with block("CREATE_ENDPOINT_CONFIG"):
        sm.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1
                }
            ]
        )

    with block("CREATE_ENDPOINT"):
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )

    with block("WAIT_FOR_ENDPOINT"):
        wait_for_endpoint(endpoint_name)

    return {
        "endpoint_name": endpoint_name,
        "endpoint_config_name": endpoint_config_name,
        "model_name": model_name,
        "instance_type": instance_type,
        "status": "IN_SERVICE"
    }
