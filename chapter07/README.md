# SageMaker AI Model Training and Tuning Capabilities

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

## Deploying the Fine-Tuned Model

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
