# Chapter 7: SageMaker AI Model Training and Tuning Capabilities

## Setting up a serverless MLFlow App

```
%pip install --upgrade boto3
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
```

```
mlflow_s3_bucket = f"mlflow-s3-bucket-{unique}"
```

```
!aws s3 mb s3://{mlflow_s3_bucket}
```

```
mlflow_app_name = f"mlflow-app-{unique}"
```

```
import boto3
client = boto3.client("sagemaker")
```

```
help(client.create_mlflow_app)
```

```
from sagemaker.core.helper.session_helper import (
    get_execution_role
)

role = get_execution_role()
```

```
registration_mode = "AutoModelRegistrationEnabled"

response = client.create_mlflow_app(
    Name=mlflow_app_name,
    ArtifactStoreUri="s3://" + mlflow_s3_bucket,
    RoleArn=role,
    ModelRegistrationMode=registration_mode
)
```

```
apps = client.list_mlflow_apps()

newly_created_app = max(
    apps["Summaries"],
    key=lambda x: x["CreationTime"]
)

app_arn = newly_created_app["Arn"]
app_arn
```

```
client.describe_mlflow_app(Arn=app_arn)
```

```
from time import sleep

def wait_for_mlflow_app(arn):
    while True:
        response = client.describe_mlflow_app(
            Arn=app_arn
        )
        status = response['Status']
        print(f"STATUS: {status}")

        if status != 'Creating':
            break

        sleep(15)
```

```
wait_for_mlflow_app(app_arn)
```

```
app_arn
```

## Fine-tuning an LLM on Amazon SageMaker AI

```
%pip uninstall -y sagemaker sagemaker-serve
%pip install "sagemaker==3.5.0"
%pip install "sagemaker-serve==1.2.0"
```

```
%pip install conda-pack
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
s3_bucket = f"mle-on-aws-2nd-ed-{unique}"
```

```
s3_bucket
```

```
!aws s3 mb s3://{s3_bucket}
```

```
import urllib.request
from pathlib import Path

URL_PARTS = [
    "https://raw.githubusercontent.com/",
    "PacktPublishing/",
    "Machine-Learning-Engineering-on-",
    "AWS-Second-Edition/",
    "refs/heads/main/",
    "chapter07/data.jsonl",
]

url = "".join(URL_PARTS)
output_file = Path("data.jsonl")
urllib.request.urlretrieve(url, output_file)
```

```
!head data.jsonl
```

```
!aws s3 cp data.jsonl s3://{s3_bucket}/data/data.jsonl
```

```
from sagemaker.core.resources import (
    ModelPackage, 
    ModelPackageGroup
)

group_name = f"model-package-group-{unique}"
model_package_group=ModelPackageGroup.create(
    model_package_group_name=group_name
)
```

```
dataset = f"s3://{s3_bucket}/data/data.jsonl"
```

```
output_path = f"s3://{s3_bucket}/output/"
```

```
import mlflow 
tracking_server = "<TRACKING ARN>" 
mlflow.set_tracking_uri(tracking_server)
```

```
mlflow_experiment_name = f"sft-experiment-{unique}"
mlflow_run_name = f"run-{unique}"
```

```
model = "meta-textgeneration-llama-3-2-1b-instruct"
```

```
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType

trainer = SFTTrainer(
    model=model, 
    training_type=TrainingType.LORA, 
    model_package_group=model_package_group,
    training_dataset=dataset, 
    s3_output_path=output_path,
    accept_eula=True ,
    mlflow_experiment_name=mlflow_experiment_name,
    mlflow_run_name=mlflow_run_name
)
```

```
trainer.__dict__
```

```
from pprint import pprint
pprint(trainer.hyperparameters.to_dict())
```

```
%%time

job = trainer.train(
    wait=True
)
```

```
pprint(job.__dict__)
```

```
from sagemaker.serve import ModelBuilder

model_builder = ModelBuilder(
    model=job, 
    instance_type="ml.g5.4xlarge"
)

model = model_builder.build()
model
```

```
pprint(model.__dict__)
```

```
artifacts = job.__dict__['model_artifacts']
artifacts
```

```
ad = artifacts.__dict__
s3_model_artifacts = ad['s3_model_artifacts']
s3_model_artifacts
```

```
!aws s3 ls {s3_model_artifacts} --recursive
```

```
merged = f"{s3_model_artifacts}/checkpoints/hf_merged"

!aws s3 ls {merged} --recursive
```

```
model_files_directory = f"model_files_{unique}"
```

```
!mkdir -p {model_files_directory}
```

```
%%time
!aws s3 cp {s3_model_artifacts}/checkpoints/hf_merged {model_files_directory}/ --recursive
```

```
%%time
!tar -czvf model.tar.gz -C {model_files_directory} .
```

```
%%time
s3_model_path = f"s3://{s3_bucket}/model/model.tar.gz"
!aws s3 cp model.tar.gz {s3_model_path}
```

```
%store s3_model_path
```

```
%store s3_bucket
```

```
containers = model.__dict__['containers']
container_image = containers[0].__dict__['image']
container_image
```

```
%store container_image
```

```
%store model_files_directory
```

```
from sagemaker.core.helper.session_helper import (
    get_execution_role
)

role = get_execution_role()
```

```
%store role
```

```
%store unique
```

## Deploying the Fine-Tuned Model

```
%store -r unique
%store -r role
%store -r container_image
%store -r model_files_directory
%store -r s3_model_path
%store -r s3_bucket
```

```
import boto3
sm = boto3.client("sagemaker")
model_name = f"model-{unique}"

sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": container_image,
        "ModelDataUrl": s3_model_path,
        "Environment": {
            "SHM_SIZE": "1g",
             "HF_MODEL_ID": "/opt/ml/model",
            "MAX_INPUT_LENGTH": "2048",
            "MAX_TOTAL_TOKENS": "4096",
            "NUM_SHARD": "1"
        }
    },
    ExecutionRoleArn=role
)
```

```
endpoint_config_name = f"endpoint-config-{unique}"

sm.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,        
            "InitialInstanceCount": 1,
            "InstanceType": "ml.g5.4xlarge",
            "InitialVariantWeight": 1
        }
    ]
)
```

```
endpoint_name = f"endpoint-{unique}"

sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
```

```
from time import sleep

def wait_for_endpoint(endpoint_name, 
                      sagemaker_client=sm):
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
wait_for_endpoint(endpoint_name)
```

```
%store endpoint_name
```

```
from pprint import pprint

response = sm.describe_model(
    ModelName=model_name
)

pprint(response)
```

```
response = sm.describe_endpoint_config(
    EndpointConfigName=endpoint_config_name
)

pprint(response)
```

```
response = sm.describe_endpoint(
    EndpointName=endpoint_name
)

pprint(response)
```

```
%store -r endpoint_name
```

```
import boto3
import json

client = boto3.client("sagemaker-runtime")

def invoke(prompt, client=client):
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 500}
        })
    )
    
    print(response["Body"].read().decode())
```

```
prompt = """
### Instruction:
Improve the given scientific sentence and provide detailed explanations to the related questions.

### Input:
- Task 1: Improve the following sentence for clarity and scientific tone:
  "It is notable to mention that various studies have shown that water serves a key role in the synthesis of vital organic compounds."

- Task 2: Explain the process through which water contributes to the synthesis of essential organic compounds.

- Task 3: Discuss how the scarcity or abundance of water in an environment impacts the diversity and functionality of life forms, particularly in terms of metabolic processes and survival strategies.

### Response:
"""

output = invoke(prompt)
output
```

```
prompt = """
### Instruction:
Write a job description.

### Input:
Role: Part-time Finance Manager  
Industry: Charity  
Location: UK  
Requirements: Previous charity experience is essential  

### Response:
"""

output = invoke(prompt)
output
```

```
prompt = """
### Instruction:
Write a comprehensive blog post and address the related topics in detail.

### Input:
- Topic: The importance of maintaining physical fitness during remote work
- Requirements:
  - Include tips and exercises that can be done at home without any special equipment
- Additional Questions:
  1. Discuss the potential benefits and drawbacks of high-intensity interval training (HIIT) for remote workers.
  2. Elaborate on the recovery process after a HIIT workout for remote workers and how they can manage it without affecting productivity.

### Response:
"""

output = invoke(prompt)
output
```

```
import boto3

sm = boto3.client("sagemaker")

sm.delete_endpoint(
    EndpointName=endpoint_name
)
```

## Performing Hyperparameter Tuning with Amazon SageMaker AI

```
```

```
```

```
```

```
```

```
```

```
```

## Deploying the Best-Performing Model from Hyperparameter Tuning

```
```

```
```

```
```

```
```

```
```

```
```
