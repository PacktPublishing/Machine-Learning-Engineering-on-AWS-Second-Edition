# Chapter 8: SageMaker AI Model Deployment Options and Strategies

In this chapter, you'll explore different deployment options and strategies in Amazon SageMaker AI. You will deploy models using real-time, serverless, asynchronous, and batch inference options, while also practicing advanced deployment techniques such as shadow testing, canary traffic shifting, and inference data capture for monitoring and evaluation.

This chapter covers the following topics:

- Preparing your JupyterLab Notebook for model deployment
- Deploying your model to a real-time inference endpoint
- Deploying your model to a serverless inference endpoint
- Running batch inference with batch transform
- Deploying your model to an asynchronous inference endpoint
- Setting up a shadow test with a SageMaker inference endpoint
- Using canary traffic shifting when performing blue/green deployments

This README.md file contains the commands and code snippets referenced in a chapter of *Machine Learning Engineering on AWS — Second Edition* by Joshua Arvin Lat, published by Packt. It is intended to support the examples in the book by making it simpler for you to copy, run, and modify the provided materials.

![Machine Learning Engineering on AWS 2nd ed](../books.png)

To help you get started more easily, the repository includes a [DETAILS.md](https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md) file containing additional guidance, references, and important notes for the examples discussed throughout the book.

## Technical Requirements

Before proceeding with the hands-on examples in this chapter, confirm that the following prerequisites and setup requirements have been addressed:

- **Sufficient account-level quota for selected ML instance types**: When deploying models, ensure that the applied account-level quota value for the selected ML instance types (for example, ml.m5.xlarge for endpoint usage , ml.m5.xlarge for transform job usage, and ml.m5.large for endpoint usage) is set to at least 2. If the quota value is 0, SageMaker AI will not be able to provision the required compute resources, and the job or endpoint creation will fail. You can review and request quota increases through the Service Quotas console (within the AWS Management Console).

- **An existing SageMaker Studio space**: You can use the SageMaker Studio space (mle-on-aws-space) that you set up in Chapter 1 to follow along with the examples in this book.

- **A code editor installed on your local machine (such as Visual Studio Code or Sublime Text)**: You'll need this when working with the code and configuration files used throughout the hands-on exercises and examples in this book.

| Note |
|:-----|
| It is recommended to use an IAM user with limited permissions instead of the root account when running the examples in this book. We will discuss this, along with other security best practices, in detail towards the end of this book. If you are just starting to use AWS, you may proceed with using the root account in the meantime. |

## Preparing your JupyterLab Notebook for Model Deployment

```
%pip uninstall -y sagemaker sagemaker-serve
%pip install "sagemaker==3.5.0"
%pip install "sagemaker-serve==1.2.0"
```

```
%pip install conda-pack
```

```
import warnings

m="Field .* has conflict with protected namespace"
warnings.filterwarnings(
    "ignore",
    message=m
)
```

```
import sagemaker
sagemaker.__file__
```

```
import os
import shutil

def copy_model_server_file():
    sagemaker_path = os.path.dirname(
        sagemaker.__file__)

    source_path = os.path.join(
        sagemaker_path, 'serve', 
        'model_server', 'triton', 'model.py')

    destination_path = os.path.join(
        sagemaker_path, 'serve', 'model.py')

    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Successfully copied "
              f"model.py from {source_path} "
              f"to {destination_path}")
    else:
        print(f"Source file not found "
              f"at {source_path}")
```

```
copy_model_server_file()
```

```
import random
import string

def generate_string(length=6):
    return ''.join(
        random.choices(string.ascii_lowercase,     
                       k=length))
```

```
unique = generate_string()
print(unique)
```

```
import boto3
region = boto3.Session().region_name
print(region)
```

```
import urllib.request
from pathlib import Path

BASE_URL_PARTS = [
    "https://raw.githubusercontent.com/",
    "PacktPublishing/",
    "Machine-Learning-Engineering-on-",
    "AWS-Second-Edition/",
    "refs/heads/main/",
    "chapter08/",
]

def download_file(filename: str):
    url = "".join(BASE_URL_PARTS) + filename
    
    output_file = Path(filename)
    urllib.request.urlretrieve(url, output_file)
    
    return output_file
```

```
download_file("model.tar.gz")
```

```
from sagemaker.core.helper.session_helper import (
    Session, 
    get_execution_role
)

session = Session()
role = get_execution_role()
bucket = session.default_bucket()
```

```
MODEL_DIRECTORY = "model"

s3_model = session.upload_data(
    "model.tar.gz", 
    bucket=bucket, 
    key_prefix="{}/{}".format(unique, MODEL_DIRECTORY)
)
```

```
s3_model
```

```
from sagemaker.core import image_uris

image = image_uris.retrieve(
    framework="xgboost", 
    region="us-east-1", 
    image_scope="inference",
    version="1.7-1"
)

print(image)
```

## Deploying your Model to a Real-Time Inference Endpoint

```
from sagemaker.serve.builder.schema_builder import (
    SchemaBuilder
)
from sagemaker.serve.spec.inference_spec import (
    InferenceSpec
)
from sagemaker.serve import ModelBuilder
```

```
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier


class XGBoostModelSpec(InferenceSpec):
    def load(self, model_dir: str):
        m = XGBClassifier()
        m.load_model(model_dir + "/xgboost-model")
        return m

    def invoke(self, 
               input_object: object, 
               model: object):
        pred_proba = model.predict_proba(input_object)
        pred = np.argmax(pred_proba, axis=1)
        return pred
```

```
inference_spec = XGBoostModelSpec()
```

```
import numpy as np

sample_input = np.array([ 1.9127373 , -0.42719723])
sample_output = np.array([1.0])

schema_builder = SchemaBuilder(
    sample_input=sample_input,
    sample_output=sample_output
)
```

```
model_builder = ModelBuilder(
    model_path=s3_model,
    role_arn=role,
    image_uri=image,
    inference_spec=inference_spec,
    schema_builder=schema_builder
)
```

```
model_builder.build()
```

```
endpoint_name = f"xgb-ep-{unique}"
```

```
predictor = model_builder.deploy(
    endpoint_name=endpoint_name
)
```

```
from sagemaker.core.deserializers import (
    JSONDeserializer
)

from sagemaker.core.serializers import (
    CSVSerializer,
)

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()
```

```
import json

def invoke_endpoint(a, b, predictor=predictor):
    payload = f"{a},{b}"

    result = predictor.invoke(
        body=payload,
        content_type="text/csv"
    )

    return int(result.body)
```

```
invoke_endpoint(-1.0, 1.5)
```

```
invoke_endpoint(1.0, -1.0)
```

```
predictor.delete()
```

## Deploying your Model to a Serverless Inference Endpoint

```
endpoint_name = f"xgb-serverless-ep-{unique}"
```

```
from sagemaker.core.inference_config import (
    ServerlessInferenceConfig
)

inference_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048
)

serverless_predictor = model_builder.deploy(
    endpoint_name=endpoint_name,
    inference_config=inference_config,
)
```

```
serverless_predictor.serializer = CSVSerializer()
serverless_predictor.deserializer = JSONDeserializer()
```

```
invoke_endpoint(
    -1.0, 1.5, 
    predictor=serverless_predictor
)
```

```
invoke_endpoint(
    1.0, -1.0, 
    predictor=serverless_predictor
)
```

```
serverless_predictor.delete()
```

## Running Batch Inference with Batch Transform

```
download_file("batch_input.csv")
```

```
INPUT_DIRECTORY = "input"

batch_input = session.upload_data(
    "batch_input.csv", 
    bucket=bucket, 
    key_prefix="{}/{}".format(unique, INPUT_DIRECTORY)
)
```

```
model_name = f"xgb-transform-model-{unique}"
model_builder.build(model_name=model_name)
```

```
output_path = f"s3://{bucket}/transform-output/"

from sagemaker.core.transformer import Transformer

transformer = Transformer(
    model_name=model_name,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    accept="text/csv",
    assemble_with="Line",
    output_path=output_path,
    sagemaker_session=session,
)
```

```
transformer.transform(
    batch_input,
    content_type="text/csv",
    split_type="Line",
    input_filter="$[1:]",
)
```

```
!aws s3 ls {output_path}
```

```
output_file_path = output_path + "batch_input.csv.out"
!aws s3 cp {output_file_path} batch_output.csv
```

```
!cat batch_output.csv
```

## Deploying your Model to an Asynchronous Inference Endpoint

```
endpoint_name = f"xgb-async-ep-{unique}"
output_path = f"s3://{bucket}/async-output/"
```

```
from sagemaker.serve.async_inference import (
    async_inference_config
)

config = async_inference_config.AsyncInferenceConfig(
    output_path=output_path
)

model_builder = ModelBuilder(
    model_path=s3_model,
    role_arn=role,
    image_uri=image,
    inference_spec=inference_spec,
    schema_builder=schema_builder
)

model_builder.build()
```

```
async_predictor = model_builder.deploy(
    endpoint_name=endpoint_name,
    inference_config=config
)
```

```
download_file("async_input.csv")
```

```
INPUT_DIRECTORY = "input"

async_input = session.upload_data(
    "async_input.csv", 
    bucket=bucket, 
    key_prefix="{}/{}".format(unique, INPUT_DIRECTORY)
)
```

```
import json

result = async_predictor.invoke_async(
    input_location=async_input,
    content_type="text/csv"
)
```

```
import boto3

client = boto3.client("logs")

ep_name = endpoint_name
log_group = f"/aws/sagemaker/Endpoints/{ep_name}"

response = client.describe_log_streams(
    logGroupName=log_group,
    orderBy="LastEventTime",
    descending=True,
    limit=1
)

latest_stream = response["logStreams"][0]

events = client.get_log_events(
    logGroupName=log_group,
    logStreamName=latest_stream["logStreamName"],
    limit=50,
    startFromHead=False
)

for e in events["events"]:
    print(e["message"])
```

```
output_location = result.output_location
output_location
```

```
!aws s3 cp {output_location} async_output.csv
```

```
!cat async_output.csv
```

```
async_predictor.delete()
```

## Setting up a Shadow Test with a SageMaker Inference Endpoint

### Creating a shadow test

```
%pip uninstall -y sagemaker sagemaker-serve
%pip install "sagemaker==3.5.0"
%pip install "sagemaker-serve==1.2.0"
```

```
from sagemaker.core.image_uris import (
    retrieve as retrieve_image
)

region = 'us-east-1'
image01_params = {
    "framework": 'huggingface', 
    "region": region, 
    "version": '4.12.3', 
    "image_scope": 'inference', 
    "base_framework_version": 'pytorch1.9.1', 
    "py_version": 'py38', 
    "container_version": 'ubuntu20.04', 
    "instance_type": 'ml.m5.xlarge'
}

image_01_uri = retrieve_image(**image01_params)
image_01_uri
```

```
image02_params = {
    "framework": 'huggingface', 
    "region": region, 
    "version": '4.17.0', 
    "image_scope": 'inference', 
    "base_framework_version": 'pytorch1.10.2', 
    "py_version": 'py38', 
    "container_version": 'ubuntu20.04', 
    "instance_type": 'ml.m5.xlarge'
}

image_02_uri = retrieve_image(**image02_params)
image_02_uri
```

```
model_01_id = 'distilbert-base-uncased-distilled-squad'

first_model = {
    'Image': image_01_uri,
    'ContainerHostname': 'firstModel',
    'Environment': {
        'HF_MODEL_ID': model_01_id,
        'HF_TASK':'question-answering'
    }
}
```

```
model_02_id = 'deepset/roberta-base-squad2'

second_model = {
    'Image': image_02_uri,
    'ContainerHostname': 'secondModel',
    'Environment': {
        'HF_MODEL_ID': model_02_id,
        'HF_TASK': 'question-answering'
    }
}
```

```
import boto3

from sagemaker.core.helper.session_helper import (
    get_execution_role
)

execution_role = get_execution_role()
sagemaker_client = boto3.client('sagemaker')
```

```
model_name = "shadow-testing-001"

model_config = {
    "ModelName": model_name,
    "ExecutionRoleArn": execution_role,
    "Containers": [first_model]
}

sagemaker_client.create_model(**model_config)
```

```
model_name = "shadow-testing-002"

model_config = {
    "ModelName": model_name,
    "ExecutionRoleArn": execution_role,
    "Containers": [second_model]
}

sagemaker_client.create_model(**model_config)
```

```
import random
import string

def generate_string(length=6):
    return ''.join(
        random.choices(string.ascii_lowercase,     
                       k=length))
```

```
unique = generate_string(length=12)
print(unique)
```

```
unique_s3_bucket_name = f"data-capture-{unique}"
```

```
!aws s3 mb s3://{ unique_s3_bucket_name }
```

```
s3_capture_path = f"s3://{unique_s3_bucket_name}/"

data_capture_config = {
    "EnableCapture": True,
    "InitialSamplingPercentage": 100,  
    "DestinationS3Uri": s3_capture_path,
    "CaptureOptions": [
        {"CaptureMode": "Input"},  
        {"CaptureMode": "Output"}  
    ],
    "CaptureContentTypeHeader": {
        "CsvContentTypes": ["text/csv"],
        "JsonContentTypes": ["application/json"]
    }
}
```

```
config_name = "shadow-testing-endpoint-config-001"

production_variants = [
    {
        "VariantName": "shadow-testing-variant-001",
        "ModelName": "shadow-testing-001",
        "InstanceType": "ml.m5.xlarge",
        "InitialInstanceCount": 1,
        "InitialVariantWeight": 1,
    }
]

shadow_variants = [
    {
        "VariantName": "shadow-testing-variant-002",
        "ModelName": "shadow-testing-002",
        "InstanceType": "ml.m5.xlarge",
        "InitialInstanceCount": 1,
        "InitialVariantWeight": 1,
    }
]
```

```
sagemaker_client.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=production_variants,
    ShadowProductionVariants=shadow_variants,
    DataCaptureConfig=data_capture_config
)
```

```
sagemaker_client.create_endpoint(
    EndpointName="shadow-testing-endpoint-001",
    EndpointConfigName=config_name,
)
```

```
from time import sleep

def wait_for_endpoint(endpoint_name):
    while True:
        response = sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        status = response['EndpointStatus']
        print(f"ENDPOINT STATUS: {status}")

        if status == 'InService':
            break
        elif status == 'Failed':
            error = "ENDPOINT CREATION FAILED"
            raise RuntimeError(error)

        sleep(15)
```

```
%%time
wait_for_endpoint("shadow-testing-endpoint-001")
```

```
sagemaker_client.describe_endpoint(
    EndpointName="shadow-testing-endpoint-001"
)
```

```
from json import loads as json_loads
from json import dumps as json_dumps

runtime_client = boto3.client('sagemaker-runtime')

def invoke_endpoint(input_payload, endpoint_name):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json_dumps(input_payload),
    )
    
    response_body = response['Body']
    
    output = json_loads(response_body.read().decode())
    return output
```

```
input_payload_01 = {"inputs": {
    "context": "Python is a widely-used programming language known for its readability and ease of use. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python is popular in data science, web development, automation, and more.",
    "question": "What are the uses of Python?",
}}

invoke_endpoint(
    input_payload=input_payload_01,
    endpoint_name="shadow-testing-endpoint-001"
)
```

```
input_payload_02 = {"inputs": {
    "context": "The Moon is Earth's only natural satellite and the fifth-largest satellite in the Solar System. It is about one-quarter the diameter of Earth, making it the largest natural satellite relative to the size of its planet.",
    "question": "How large is the Moon?",
}}

invoke_endpoint(
    input_payload=input_payload_02,
    endpoint_name="shadow-testing-endpoint-001"
)
```

### Inspecting the captured data

```
!aws s3 ls { s3_capture_path } --recursive
```

```
!aws s3 cp { s3_capture_path } . --recursive
```

### Promoting a shadow variant

```
endpoint_config_name = "shadow-testing-endpoint-config-002"

single_variant = {
    "VariantName": "shadow-testing-variant-002",
    "ModelName": "shadow-testing-002",
    "InstanceType": "ml.m5.xlarge",
    "InitialInstanceCount": 1,
    "InitialVariantWeight": 1,
}

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        single_variant
    ],
   DataCaptureConfig=data_capture_config
)
```

```
sagemaker_client.update_endpoint(
    EndpointName="shadow-testing-endpoint-001",
    EndpointConfigName=endpoint_config_name,
)
```

```
%%time
wait_for_endpoint("shadow-testing-endpoint-001")
```

```
sagemaker_client.describe_endpoint(
    EndpointName="shadow-testing-endpoint-001"
)
```

### Cleaning up

```
sagemaker_client.delete_endpoint(
    EndpointName="shadow-testing-endpoint-001"
)
```

```
endpoint_configurations = [
    "shadow-testing-endpoint-config-001",
    "shadow-testing-endpoint-config-002"
]

for config in endpoint_configurations:
    sagemaker_client.delete_endpoint_config(
        EndpointConfigName=config
    )
```

```
models = [
    "shadow-testing-001",
    "shadow-testing-002"
]

for model in models:
    sagemaker_client.delete_model(
        ModelName=model
    )
```

## Using Canary Traffic Shifting when performing Blue/Green Deployments

### Configuring a Canary Deployment in SageMaker AI

```
print(first_model)
```

```
model_config = {
    "ModelName": "traffic-shifting-001",
    "ExecutionRoleArn": execution_role,
    "Containers": [first_model]
}
sagemaker_client.create_model(**model_config)
```

```
config_name = "traffic-shifting-endpoint-config-001"
only_variant = {
    "VariantName": "traffic-shifting-only-variant",
    "ModelName": "traffic-shifting-001",
    "InstanceType": "ml.m5.xlarge",
    "InitialInstanceCount": 2,
    "InitialVariantWeight": 1,
}

sagemaker_client.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[only_variant],
    DataCaptureConfig=data_capture_config
)
```

```
deployment_config = {
    "BlueGreenUpdatePolicy": {
        "TrafficRoutingConfiguration": {
            "Type": "CANARY",
            "CanarySize": {
                "Type": "CAPACITY_PERCENT",
                "Value": 40
            },
            "WaitIntervalInSeconds": 1200
        },
        "TerminationWaitInSeconds": 1200,
        "MaximumExecutionTimeoutInSeconds": 3600
    },
}
```

```
sagemaker_client.create_endpoint(
    EndpointName="traffic-shifting-endpoint-001",
    EndpointConfigName=config_name,
    DeploymentConfig=deployment_config
)
```

```
%%time
wait_for_endpoint("traffic-shifting-endpoint-001")
```

```
sagemaker_client.describe_endpoint(
    EndpointName="traffic-shifting-endpoint-001"
)
```

```
print(input_payload_01)
```

```
invoke_endpoint(
    input_payload=input_payload_01,
    endpoint_name="traffic-shifting-endpoint-001"
)
```

```
print(input_payload_02)
```

```
invoke_endpoint(
    input_payload=input_payload_02,
    endpoint_name="traffic-shifting-endpoint-001"
)
```

```
print(second_model)
```

```
model_config = {
    "ModelName": "traffic-shifting-002",
    "ExecutionRoleArn": execution_role,
    "Containers": [second_model]
}
sagemaker_client.create_model(**model_config)
```

```
config_name = "traffic-shifting-endpoint-config-002"

new_variant = {
    "VariantName": "traffic-shifting-only-variant",
    "ModelName": "traffic-shifting-002",
    "InstanceType": "ml.m5.xlarge",
    "InitialInstanceCount": 2,
    "InitialVariantWeight": 1,
}

sagemaker_client.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[new_variant],
    DataCaptureConfig=data_capture_config
)
```

```
sagemaker_client.update_endpoint(
    EndpointName="traffic-shifting-endpoint-001",
    EndpointConfigName=config_name,
    RetainDeploymentConfig=True
)
```

```
from pprint import pprint

def wait_and_describe_endpoint(endpoint_name):
    while True:
        response = sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        
        status = response['EndpointStatus']
        
        print("-"*100)
        print(f"ENDPOINT STATUS: {status}")
        print("-"*100)
        pprint(response, indent=2)
        print("-"*100)

        if status == 'InService':
            break
        elif status == 'Failed':
            error = "ENDPOINT CREATION FAILED"
            raise RuntimeError(error)

        sleep(120)
```

```
%%time
endpoint_name = "traffic-shifting-endpoint-001"
wait_and_describe_endpoint(endpoint_name)
```

```
print(input_payload_01)
```

```
invoke_endpoint(
    input_payload=input_payload_01,
    endpoint_name="traffic-shifting-endpoint-001"
)
```

```
print(input_payload_02)
```

```
invoke_endpoint(
    input_payload=input_payload_02,
    endpoint_name="traffic-shifting-endpoint-001"
)
```

### Cleaning up

```
sagemaker_client.delete_endpoint(
    EndpointName="traffic-shifting-endpoint-001"
)
```

```
endpoint_configurations = [
    "traffic-shifting-endpoint-config-001",
    "traffic-shifting-endpoint-config-002",
]

for config in endpoint_configurations:
    sagemaker_client.delete_endpoint_config(
        EndpointConfigName=config
    )
```

```
models = [
    "traffic-shifting-001",
    "traffic-shifting-002"
]

for model in models:
    sagemaker_client.delete_model(
        ModelName=model
    )
```

## Where to Get Your Copy

You can grab your copy of *Machine Learning Engineering on AWS — Second Edition* through popular online retailers such as [Amazon](https://amazon.com/author/arvs) or directly from the publisher, [Packt](https://www.packtpub.com/en-us/product/machine-learning-engineering-on-aws-9781835881088). Feel free to choose the format that works best for you. 🙏

## Get to Know the Author

**Joshua Arvin Lat** serves as the Vice President of Cybersecurity and AI for **Axos**. He previously held Chief Technology Officer and Director roles across SaaS platforms, AI automation companies, e-commerce startups, and digital agencies. Because of his proven track record in leading digital transformation within organizations, he has been recognized as one of the winners of the prestigious Orange Boomerang: Digital Leader of the Year 2023 award. 

![Machine Learning Engineering on AWS 2nd ed](../arvs-machine-learning-engineering-on-aws.png)

Years ago, he led a team that won first place in a global cybersecurity competition for their published research. He is also an AWS AI Hero and has spoken at several international conferences on practical applications of generative AI, software engineering, cybersecurity, and management.

## Other Books by the Author

You can find the author's other books on AI and Cybersecurity by visiting the [Amazon Author Page](https://amazon.com/author/arvs)

![Other 4 books](../previous-books.png)

___

<table>
  <tr>
    <td>PREVIOUS</td>
    <td>
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter07">
        https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter07
      </a>
    </td>
  </tr>
  <tr>
    <td>NEXT</td>
    <td>
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter09">
        https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter09
      </a>
    </td>
  </tr>
</table>
