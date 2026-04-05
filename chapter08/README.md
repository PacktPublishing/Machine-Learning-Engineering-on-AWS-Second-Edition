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
