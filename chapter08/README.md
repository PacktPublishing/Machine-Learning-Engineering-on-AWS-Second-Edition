# SageMaker AI Model Deployment Options and Strategies

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
```

```
```

## Deploying your Model to a Serverless Inference Endpoint

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

```
```

```
```

## Running Batch Inference with Batch Transform

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

```
```

```
```

## Deploying your Model to an Asynchronous Inference Endpoint

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

```
```

```
```

## Setting up a Shadow Test with a SageMaker Inference Endpoint

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

```
```

```
```

## Using Canary Traffic Shifting when performing Blue/Green Deployments

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

```
```

```
```
