# Chapter 3: Machine Learning Engineering with Amazon SageMaker AI

## Setting up and preparing your JupyterLab notebook

```
%pip uninstall -y sagemaker sagemaker-serve
%pip install "sagemaker==3.2.0"
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
import boto3

region = boto3.Session().region_name
print(region)
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
    "chapter01/import_helper.py",
]

url = "".join(URL_PARTS)
output_file = Path("import_helper.py")
urllib.request.urlretrieve(url, output_file)
```

```
import sagemaker
from import_helper import ImportHelper

helper = ImportHelper(sagemaker)
helper.explore()
```

```
helper.guess_import("SchemaBuilder")
```

```
im = ["InferenceSpec", 
      "Session", 
      "ModelBuilder", 
      "ModelTrainer"]

for i in im:
    print("-"*20)
    print(f"Running helper.guess_import() for {i}")
    helper.guess_import(i)
```

```
from sagemaker.serve.builder.schema_builder import (
    SchemaBuilder
)
from sagemaker.serve.spec.inference_spec import (
    InferenceSpec
)
from sagemaker.core.helper.session_helper import (
    Session, 
    get_execution_role
)
from sagemaker.serve import ModelBuilder
from sagemaker.train import ModelTrainer
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

## Preparing a Synthetic Dataset for Binary Classification

```
var = {}
```

```
import pandas as pd
from sklearn.datasets import make_moons

def generate_dataset(samples=1000, 
                     noise=0.4,  
                     random_state=200):
    params = {
        'n_samples': samples, 
        'noise': noise,
        'random_state': random_state
    }
    X, y = make_moons(**params)
    
    return pd.DataFrame(
        dict(target=y, a=X[:,0], b=X[:,1])
    )
```

```
var['dataset'] = generate_dataset()
var['dataset']
```

```
from matplotlib import pyplot

def plot_dataset(dataset, 
                 colors = {0:'red', 1:'blue'}):
    fig, ax = pyplot.subplots()
    grouped = dataset.groupby('target')
    
    for key, group in grouped:
        params = {
            'ax': ax, 
            'kind': 'scatter', 
            'x': 'a', 
            'y': 'b',
            'label': key, 
            'color': colors[key]
        }
        
        group.plot(**params)

    pyplot.show()
```

```
plot_dataset(var['dataset'])
```

```
from sklearn.model_selection import train_test_split

def split_data(dataset, 
               test_size=0.2, 
               random_state=0):

    AB, C = train_test_split(
        dataset, 
        test_size=test_size, 
        random_state=random_state
    )

    A, B = train_test_split(
        AB, 
        test_size=0.25, 
        random_state=random_state
    )
    
    return A, B, C
```

```
training, validation, test = split_data(
    var['dataset']
)

var['training'] = training
var['validation'] = validation
var['test'] = test
```

```
var['training']
```

```
!mkdir -p data
```

```
var['training'].to_csv("./data/train.csv", 
                       index=False, 
                       header=False)
```

```
var['validation'].to_csv("./data/validation.csv",
                         index=False, 
                         header=False)
```

```
var['test'].to_csv("./data/test.csv", 
                   index=False, 
                   header=False)
```

```
test_data = var['test'].drop("target", axis=1)
test_data
```

```
test_data.to_csv("./data/test_no_target.csv", 
                 index=False, 
                 header=False)
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
prefix = generate_string()
print(prefix)
```

```
session = Session()
role = get_execution_role()
bucket = session.default_bucket()
```

```
TRAIN_DATA = "train.csv"
VALIDATION_DATA = "validation.csv"
TEST_DATA = "test.csv"
DATA_DIRECTORY = "data"
```

```
train_input = session.upload_data(
    DATA_DIRECTORY, 
    bucket=bucket, 
    key_prefix="{}/{}".format(prefix, DATA_DIRECTORY)
)
```

```
!aws s3 ls {train_input} --recursive
```

```
s3_train_input_path = "s3://{}/{}/data/{}".format(
    bucket, 
    prefix, 
    TRAIN_DATA
)

s3_val_input_path = "s3://{}/{}/data/{}".format(
    bucket, 
    prefix, 
    VALIDATION_DATA
)

s3_output_path = "s3://{}/{}/output".format(
    bucket, 
    prefix
)

s3_test_path = "s3://{}/{}/data/{}".format(
    bucket, 
    prefix, 
    TEST_DATA
)
```

## Training an XGBoost binary classifier

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
    "num_round": "50",
    "eval_metric": "error",
}
```

```
base_job_name = "xgb-" + prefix
```

```
model_trainer = ModelTrainer(
    base_job_name=base_job_name,
    hyperparameters=hyperparameters,
    training_image=image,
    training_input_mode="File",
    role=role,
    output_data_config=OutputDataConfig(
        s3_output_path=s3_output_path
    ),
    stopping_condition=StoppingCondition(
        max_runtime_in_seconds=600
    ),
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
model_trainer.train(
    input_data_config=[
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
)
```

## Deploying an XGBoost model to a real-time inference endpoint

```
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
```

```
test_data
```

```
sample_input = test_data.iloc[0].to_numpy()
sample_input
```

```
sample_output = np.array([1.0])
```

```
schema_builder = SchemaBuilder(
    sample_input=sample_input,
    sample_output=sample_output
)
```

```
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
instance_type = "ml.m5.xlarge"
```

```
model_builder = ModelBuilder(
    model=model_trainer,
    role_arn=role,
    image_uri=image,
    inference_spec=inference_spec,
    schema_builder=schema_builder,
    instance_type=instance_type,
)
```

```
model_builder.build()
```

```
endpoint_name = f"xgb-{prefix}-ep"

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

result = predictor.invoke(
    body=[-1.0, 1.5],
    content_type="text/csv"
)

result.body
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
predictions = []

for _, row in var['test'].iterrows():
    pred = invoke_endpoint(row['a'], row['b'])
    predictions.append(pred)


var['test'].insert(0, 'predicted_value', predictions)
```

```
var['test']
```

```
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

target = var['test']['target']
predicted = var['test']['predicted_value']

accuracy = accuracy_score(target, predicted)
precision = precision_score(target, predicted)
recall = recall_score(target, predicted)
f1 = f1_score(target, predicted)
conf_matrix = confusion_matrix(target, predicted)
```

```
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
```

```
predictor.delete()
```

## Setting up BERT fine-tuning with SageMaker JumpStart
## Using a smaller dataset for fine-tuning
## Running the BERT model fine-tuning job
## Deploying the fine-tuned model to a real-time inference endpoint
