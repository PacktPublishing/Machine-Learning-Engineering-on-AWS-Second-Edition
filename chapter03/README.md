<img width="468" height="473" alt="image" src="https://github.com/user-attachments/assets/bed16e13-e34a-4aa8-b931-94025b5eb6ec" /><img width="468" height="23" alt="image" src="https://github.com/user-attachments/assets/a2bf0164-2190-41b2-8c97-3c9717319129" /># Chapter 3: Machine Learning Engineering with Amazon SageMaker AI

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
## Training an XGBoost binary classifier
## Deploying an XGBoost model to a real-time inference endpoint
## Setting up BERT fine-tuning with SageMaker JumpStart
## Using a smaller dataset for fine-tuning
## Running the BERT model fine-tuning job
## Deploying the fine-tuned model to a real-time inference endpoint
