```
from sagemaker.mlops.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum
)

from sagemaker.core.workflow.conditions import (
    ConditionEquals
)

from sagemaker.mlops.workflow import (
    ConditionStep
)

from sagemaker.core.workflow.functions import (
    JsonGet
)

from sagemaker.core.lambda_helper import (
    Lambda
)
```

```
ENDPOINT_NAME = "endpoint-100"

endpoint_name_param = ParameterString(
    name="EndpointName",
    default_value=ENDPOINT_NAME
)
```

```
json_reader_lambda = LambdaStep(
    name="ReadS3JSON",
    lambda_func=Lambda(
        function_arn="<SPECIFY LAMBDA ARN>"
    ),
    inputs={
        "s3_path": Join(
            on="", 
            values=[ts3uri, "results.json"]
        ),
    },
    outputs=[
        LambdaOutput(
            output_name="model_arn",
            output_type=LambdaOutputTypeEnum.String
        )
    ],
    depends_on=[eval_step]
)
```

```
endpoint_exists_lambda = LambdaStep(
    name="CheckIfEndpointExists",
    lambda_func=Lambda(
        function_arn="<SPECIFY LAMBDA ARN>"
    ),
    inputs={
        "endpoint_name": endpoint_name_param
    },
    outputs=[
        LambdaOutput(
            output_name="endpoint_exists",
            output_type=LambdaOutputTypeEnum.Boolean
        )
    ],
    depends_on=[json_reader_lambda]
)
```

```
properties_1 = json_reader_lambda.properties
json_model_arn_1 = properties_1.Outputs['model_arn']

step_lambda_deploy_to_existing_endpoint = LambdaStep(
    name="DeployToExistingEndpoint",
    lambda_func=Lambda(
        function_arn="<SPECIFY LAMBDA ARN>"
    ),
    inputs={
        "endpoint_name": endpoint_name_param,
        "model_arn": json_model_arn_1,
        "instance_type": "ml.g5.4xlarge"
    },
    depends_on=[endpoint_exists_lambda]
)
```

```
properties_2 = json_reader_lambda.properties
json_model_arn_2 = properties_2.Outputs['model_arn']

step_lambda_deploy_to_new_endpoint = LambdaStep(
    name="DeployToNewEndpoint",
    lambda_func=Lambda(
        function_arn="<SPECIFY LAMBDA ARN>"
    ),
    inputs={
        "endpoint_name": endpoint_name_param,
        "model_arn": json_model_arn_2,
        "instance_type": "ml.g5.4xlarge"
    },
    depends_on=[endpoint_exists_lambda]
)
```

```
Outputs = endpoint_exists_lambda.properties.Outputs
left = Outputs['endpoint_exists']

cond_equals = ConditionEquals(
    left=left,
    right=True
)

if_steps = [step_lambda_deploy_to_existing_endpoint]
else_steps = [step_lambda_deploy_to_new_endpoint]

step_endpoint_exists_condition = ConditionStep(
    name="EndpointExists",
    conditions=[cond_equals],
    if_steps=if_steps,
    else_steps=else_steps
)
```

```
pipeline = Pipeline(
    name=f"CompletePipeline-{unique}",
    steps=[
        train_step,
        eval_step,
        json_reader_lambda,
        endpoint_exists_lambda,
        step_endpoint_exists_condition
    ],
    parameters=[
        role_arn_param,
        train_dataset_param,
        train_output_param,
        mlflow_exp_param,
        mlflow_run_param,
        region_param,
        eval_dataset_param,
        custom_metric_param,
        eval_output_param,
        endpoint_name_param
    ],
    sagemaker_session=pipeline_session,
)
```

```
pipeline.upsert(role_arn=role)
```

```
execution = pipeline.start(
    parameters={
        "Dataset": train_dataset_s3,
        "TrainingOutputPath": train_output_s3,
        "MLflowExperimentName": (
            f"sft-experiment-{unique}"
        ),
        "MLflowRunName": (
            f"run-{unique}"
        ),
        "Region": region,
        "TrainingRoleArn": role,
        "EvalDataset": eval_dataset_s3,
        "CustomMetricPath": custom_metric_s3,
        "EvaluationOutputPath": eval_output_s3,
        "EndpointName": ENDPOINT_NAME
    }
)
```

```
%%time
import time

while True:
    details = execution.describe()
    status = details["PipelineExecutionStatus"]
    print("Status:", status)

    if status != "Executing":
        break

    time.sleep(60)
```

```
import boto3
import json

client = boto3.client("sagemaker-runtime")

def invoke(prompt, client=client):
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
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
import boto3

sm = boto3.client("sagemaker")

sm.delete_endpoint(
    EndpointName=ENDPOINT_NAME
)
```
