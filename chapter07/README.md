# Chapter 7: SageMaker AI Model Training and Tuning Capabilities

In this chapter, you'll fine-tune a large language model using Amazon SageMaker AI as part of an end-to-end machine learning workflow. You will track experiments using MLflow, execute supervised fine-tuning jobs, perform hyperparameter tuning to identify the best-performing model, and deploy the final model to a real-time inference endpoint.

We will cover the following topics in this chapter:

- Setting up a serverless MLflow App
- Fine-tuning an LLM on Amazon SageMaker AI
- Deploying the Fine-Tuned Model
- Performing Hyperparameter Tuning with Amazon SageMaker AI
- Deploying the Best-Performing Model from Hyperparameter Tuning

This README.md file contains the commands and code snippets referenced in a chapter of *Machine Learning Engineering on AWS — Second Edition* by Joshua Arvin Lat, published by Packt. It is intended to support the examples in the book by making it simpler for you to copy, run, and modify the provided materials.

![Machine Learning Engineering on AWS 2nd ed](../books.png)

To help you get started more easily, the repository includes a [DETAILS.md](https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md) file containing additional guidance, references, and important notes for the examples discussed throughout the book.

## Technical requirements

Before proceeding with the hands-on examples in this chapter, ensure that the following prerequisites and setup requirements are in place:

- **An existing SageMaker Studio space**: You can use the SageMaker Studio space (mle-on-aws-space) that you set up in Chapter 1. Ensure that the space is configured with sufficient storage capacity (for example, at least 100 GB) to accommodate datasets, model artifacts, logs, and intermediate training outputs generated throughout the examples in this chapter.

- **Sufficient account-level quota for selected ML instance types**: Ensure that your AWS account has the required applied account-level quota values for the SageMaker AI ML instance types used in this chapter. You should have at least 1× ml.g5.4xlarge for the real-time inference endpoint (ml.g5.4xlarge for endpoint usage), at least 3× ml.m5.xlarge training instances to support hyperparameter tuning (ml.m5.xlarge for spot training job usage), and at least 1× ml.m5.xlarge for the real-time inference endpoint (ml.m5.xlarge for endpoint usage). If any of these quotas are set to 0 or below the required level, SageMaker AI will not be able to provision the necessary compute resources, causing training, tuning, or endpoint creation to fail. You can review and adjust these limits in the Service Quotas console in the AWS Management Console.

- **A code editor installed on your local machine (such as Visual Studio Code or Sublime Text)**: You'll need this when working with the code and configuration files used throughout the hands-on exercises and examples in this book.


## Setting up a serverless MLflow App

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
%pip uninstall -y sagemaker sagemaker-serve
%pip install "sagemaker==3.5.0"
%pip install "sagemaker-serve==1.2.0"
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
TRAIN_DATA = "train.csv"
VALIDATION_DATA = "validation.csv"
TEST_DATA = "test.csv"
DATA_DIRECTORY = "data"
```

```
import urllib.request
from pathlib import Path
from typing import Union

BASE_URL_PARTS = [
    "https://raw.githubusercontent.com/",
    "PacktPublishing/",
    "Machine-Learning-Engineering-on-",
    "AWS-Second-Edition/",
    "refs/heads/main/",
    "chapter07/",
]

def download_file(
        filename: str, 
        directory: Union[str, Path] = DATA_DIRECTORY
    ) -> Path:
    url = "".join(BASE_URL_PARTS) + filename
    
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    
    output_file = directory_path / filename
    urllib.request.urlretrieve(url, output_file)
    
    return output_file
```

```
download_file(TRAIN_DATA)
download_file(VALIDATION_DATA)
download_file(TEST_DATA)
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
train_input = session.upload_data(
    DATA_DIRECTORY, 
    bucket=bucket, 
    key_prefix="{}/{}".format(unique, DATA_DIRECTORY)
)
```

```
!aws s3 ls {train_input} --recursive
```

```
s3_train_input_path = "s3://{}/{}/data/{}".format(
    bucket, 
    unique, 
    TRAIN_DATA
)

s3_val_input_path = "s3://{}/{}/data/{}".format(
    bucket, 
    unique, 
    VALIDATION_DATA
)

s3_output_path = "s3://{}/{}/output".format(
    bucket, 
    unique
)

s3_test_path = "s3://{}/{}/data/{}".format(
    bucket, 
    unique, 
    TEST_DATA
)
```

```
import os
import shutil
import sagemaker

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
from sagemaker.core.shapes import (
    Channel,
    DataSource,
    S3DataSource,
    OutputDataConfig,
    StoppingCondition,
)

from sagemaker.core import image_uris
```

```
import boto3

region = boto3.Session().region_name
print(region)
```

```
image = image_uris.retrieve(
    framework="xgboost", 
    region=region, 
    image_scope="training",
    version="1.7-1"
)

print(image)
```

```
hyperparameters = {
    "objective": "binary:logistic",
    "num_round": "50"
}
```

```
from sagemaker.core.parameter import IntegerParameter

hyperparameter_ranges = {
    "max_depth": IntegerParameter(4, 10),
}
```

```
base_job_name = "xgb-hpo-" + unique
```

```
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute

compute = Compute(
    enable_managed_spot_training=True
)

model_trainer = ModelTrainer(
    base_job_name=base_job_name,
    hyperparameters=hyperparameters,
    training_image=image,
    training_input_mode="File",
    role=role,
    compute=compute,
    output_data_config=OutputDataConfig(
        s3_output_path=s3_output_path
    ),
    stopping_condition=StoppingCondition(
        max_runtime_in_seconds=1800,
        max_wait_time_in_seconds=1800
    ),
)
```

```
from sagemaker.train.tuner import HyperparameterTuner

metric_definitions = [
    {
        "Name": "validation-logloss",
        "Regex": "validation-logloss:([0-9\\.]+)"
    },
    {
        "Name": "train-logloss",
        "Regex": "train-logloss:([0-9\\.]+)"
    }
]

tuner = HyperparameterTuner(
    model_trainer=model_trainer,
    objective_metric_name="validation:f1",
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=metric_definitions,
    max_jobs=6,
    max_parallel_jobs=3,
)
```

```
train_data_source=DataSource(
    s3_data_source=S3DataSource(
        s3_data_type="S3Prefix",
        s3_uri=s3_train_input_path,               
        s3_data_distribution_type="FullyReplicated",
    )
)

val_data_source=DataSource(
    s3_data_source=S3DataSource(
        s3_data_type="S3Prefix",
        s3_uri=s3_val_input_path,            
        s3_data_distribution_type="FullyReplicated",
    )
)
```

```
%%time

tuner.tune(
    inputs=[
        Channel(
            channel_name="train",
            content_type="csv",
            compression_type="None",
            record_wrapper_type="None",
            data_source=train_data_source,
        ),
        Channel(
            channel_name="validation",
            content_type="csv",
            compression_type="None",
            record_wrapper_type="None",
            data_source=val_data_source,
        )
    ],
    wait=True
)
```

```
tuner.describe()
```

```
analytics = tuner.analytics()
df = analytics.dataframe()
```

```
df
```

```
df.sort_values(
    by="FinalObjectiveValue", 
    ascending=False
)
```

```
best_training_job = tuner.best_training_job()
best_training_job
```

```
%store best_training_job
%store unique
%store role
```

## Deploying the Best-Performing Model from Hyperparameter Tuning

```
%store -r best_training_job
%store -r unique
%store -r role
```

```
from sagemaker.core.resources import TrainingJob

job = TrainingJob.get(best_training_job)
job
```

```
job.__dict__
```

```
algo_spec = job.__dict__['algorithm_specification']
container_image = algo_spec.training_image
print(container_image)
```

```
model_artifacts = job.__dict__['model_artifacts']
s3_model_path = model_artifacts.s3_model_artifacts
print(s3_model_path)
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
            "InstanceType": "ml.m5.xlarge",
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
from sagemaker.core.resources import Endpoint
predictor = Endpoint.get(endpoint_name)

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

result = predictor.invoke(
    body=[1.0, -1.0],
    content_type="text/csv"
)

result.body
```

```
import json

result = predictor.invoke(
    body=[-1.0, 1.5],
    content_type="text/csv"
)

result.body
```

```
import json

def invoke_endpoint(a, b, predictor=predictor):
    payload = f"{a},{b}"

    result = predictor.invoke(
        body=payload,
        content_type="text/csv"
    )

    probability = float(result.body)

    prediction = int(probability > 0.5)

    return prediction
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

## Where to Get Your Copy

You can grab your copy of *Machine Learning Engineering on AWS — Second Edition* through popular online retailers such as [Amazon](https://amazon.com/author/arvs) or directly from the publisher, [Packt](https://www.packtpub.com/en-us/product/machine-learning-engineering-on-aws-9781835881088). Feel free to choose the format that works best for you. 🙏

## Get to Know the Author

**Joshua Arvin Lat** serves as the Vice President of Cybersecurity and AI for **Axos**. He previously held Chief Technology Officer and Director roles across SaaS platforms, AI automation companies, e-commerce startups, and digital agencies. Because of his proven track record in leading digital transformation within organizations, he has been recognized as one of the winners of the prestigious Orange Boomerang: Digital Leader of the Year 2023 award. 

![Machine Learning Engineering on AWS 2nd ed](../arvs-machine-learning-engineering-on-aws.png)

Years ago, he led a team that won first place in a global cybersecurity competition for their published research. He is also an AWS AI Hero and has spoken at several international conferences on practical applications of generative AI, software engineering, cybersecurity, and management.

## Other Books by the Author

You can find the author's other books on AI and Cybersecurity by visiting the [Amazon Author Page](https://amazon.com/author/arvs)

![Other 4 books](../previous-books.png)
