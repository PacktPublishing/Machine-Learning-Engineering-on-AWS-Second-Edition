# Chapter 9: Automating LLMOps Workflows with SageMaker Pipelines

In this chapter, you'll design and operationalize LLMOps pipelines using SageMaker Pipelines. You will build single-step and multi-step workflows, integrate AWS Lambda-based orchestration steps, and learn best practices for building scalable, maintainable, secure, and cost-efficient production-grade machine learning pipelines.

This chapter covers the following topics:

- Setting up the project environment and dependencies
- Building and Running the Single-Step Fine-Tuning Pipeline
- Building and Running the Single-Step Evaluation Pipeline
- Configuring and Running a Two-Step Fine-Tuning and Evaluation Pipeline
- Preparing the Lambda functions for deploying a model to an endpoint
- Completing the LLMOps pipeline
- Best Practices and Key Considerations for Building Automated ML Workflows

This README.md file contains the commands and code snippets referenced in a chapter of *Machine Learning Engineering on AWS — Second Edition* by Joshua Arvin Lat, published by Packt. It is intended to support the examples in the book by making it simpler for you to copy, run, and modify the provided materials.

![Machine Learning Engineering on AWS 2nd ed](../books.png)

To help you get started more easily, the repository includes a [DETAILS.md](https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md) file containing additional guidance, references, and important notes for the examples discussed throughout the book.

## Technical Requirements

Before proceeding with the hands-on examples in this chapter, confirm that the following prerequisites and setup requirements have been addressed:

- **Sufficient account-level quota for selected ML instance types**: Ensure that your AWS account has the required applied account-level quota values for the SageMaker AI ML instance types used in this chapter. You should have at least 2× ml.g5.4xlarge for the real-time inference endpoint (ml.g5.4xlarge for endpoint usage), and at least 2× ml.m5.xlarge for processing job usage and pipeline execution workloads (ml.m5.xlarge for processing job usage). You can review and adjust these limits in the Service Quotas console in the AWS Management Console.

- **An existing SageMaker Studio space**: You can use the SageMaker Studio space (mle-on-aws-space) that you set up in Chapter 1 to follow along with the examples in this book. You may upgrade the space to an ml.m5.xlarge instance type and allocate a minimum of 100 GB of storage to properly support the compute and data requirements of the hands-on exercises.

- **A code editor installed on your local machine (such as Visual Studio Code or Sublime Text)**: You'll need this when working with the code and configuration files used throughout the hands-on exercises and examples in this book.

- **Updated space execution role trust relationship configuration**: Ensure that the execution role attached to your SageMaker Studio space allows both SageMaker AI and Amazon Bedrock to assume the role. This is required because the workflows and evaluation jobs used throughout this chapter rely on SageMaker AI processing jobs and Bedrock-powered LLM-as-a-Judge evaluation capabilities. The trust policy for the execution role should include the following configuration:

    ```
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": [
                        "sagemaker.amazonaws.com",
                        "bedrock.amazonaws.com"
                    ]
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    ```
    
    If the execution role does not include these trusted service principals, processing jobs, pipeline executions, or evaluation tasks may fail when attempting to access SageMaker AI or Amazon Bedrock resources. You can review and update the trust relationship configuration from the IAM console in the AWS Management Console.

- **Updated execution role permissions for SageMaker Studio space**: Ensure that the execution role attached to your SageMaker Studio space includes the required service permissions for running end-to-end LLMOps workflows. In addition to the correct trust relationship for SageMaker AI and Amazon Bedrock, the role must also include the following managed policies: AmazonBedrockFullAccess, AmazonS3FullAccess, AmazonSageMakerFullAccess, AWSLambda_FullAccess, and CloudWatchFullAccessV2. You can review and update the IAM permissions attached to the execution role from the IAM console in the AWS Management Console.

| Note |
|:-----|
| While working on the examples of this chapter, you may optionally use AdministratorAccess to reduce permission-related errors and simplify setup. Keep in mind that the IAM roles configuration used across the chapter is overly permissive and should not be used in production environments without further refinement and restriction of permissions. |

## Setting Up the Project Environment and Dependencies

```
%pip uninstall -y sagemaker sagemaker-serve
%pip install "sagemaker==3.5.0"
%pip install "sagemaker-serve==1.2.0"
%pip install "rich==14.2.0"
```

```
CODE_DIRECTORY = "code"
!mkdir -p {CODE_DIRECTORY}
```

```
DATA_DIRECTORY = "data"
!mkdir -p {DATA_DIRECTORY}
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
    "chapter09/",
]

def download_file(
        filename: str, 
        directory: Union[str, Path] = CODE_DIRECTORY
    ) -> Path:
    url = "".join(BASE_URL_PARTS) + filename
    
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    
    output_file = directory_path / filename
    urllib.request.urlretrieve(url, output_file)
    
    return output_file
```

```
download_file("fine_tuning.py")
download_file("evaluation.py")
```

```
!cat code/fine_tuning.py
```

```
!cat code/evaluation.py
```

```
download_file(
    "data.jsonl", 
    directory=DATA_DIRECTORY
)
```

```
!head data/data.jsonl
```

```
download_file(
    "custom_metric.json", 
    directory="."
)
```

```
!cat custom_metric.json
```

```
download_file(
    "gen_qa.jsonl", 
    directory=DATA_DIRECTORY
)
```

```
download_file(
    "pipeline_wrapper.py", 
    directory="."
)
```

```
!cat pipeline_wrapper.py
```

## Building and Running the Single-Step Fine-Tuning Pipeline

### Building and Running the Pipeline

```
import random
import string

from sagemaker.core.workflow.parameters import (   
    ParameterString
)
from sagemaker.core.helper.session_helper import (
    Session, 
    get_execution_role
)
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingOutput, 
    ProcessingS3Output
)
from sagemaker.mlops.workflow.pipeline import (
    Pipeline
)
from sagemaker.mlops.workflow.steps import (
    ProcessingStep
)
from sagemaker.core.workflow.pipeline_context import (
    PipelineSession
)
from sagemaker.core import image_uris
```

```
def generate_string(length=6):
    return ''.join(
        random.choices(
            string.ascii_lowercase, 
            k=length
        )
    )
```

```
unique = generate_string()
sagemaker_session = Session()
pipeline_session = PipelineSession()
role = get_execution_role()
region = sagemaker_session.boto_region_name

prefix = f"pipeline-{unique}"
s3 = sagemaker_session.default_bucket()
dataset_path = f"s3://{s3}/{prefix}/data/data.jsonl"
output_path = f"s3://{s3}/{prefix}/output/"
```

```
!aws s3 cp data/data.jsonl {dataset_path}
```

```
role_arn_param = ParameterString(
    name="TrainingRoleArn",
    default_value=role
)

dataset_param = ParameterString(
    name="Dataset",
    default_value=dataset_path
)

output_path_param = ParameterString(
    name="OutputPath",
    default_value=output_path
)

mlflow_exp_param = ParameterString(
    name="MLflowExperimentName",
    default_value=f"sft-experiment-{unique}"
)

mlflow_run_param = ParameterString(
    name="MLflowRunName",
    default_value=f"run-{unique}"
)

region_param = ParameterString(
    name="Region",
    default_value=region
)
```

```
processor = ScriptProcessor(
    image_uri=image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    ),
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name=f"sfttrainer-job-{unique}",
    sagemaker_session=pipeline_session,
    role=role,
)
```

```
processing_output_path = f"s3://{s3}/{prefix}/output"
local_path = "/opt/ml/processing/output"
```

```
step_args = processor.run(
    inputs=[],
    outputs=[
        ProcessingOutput(
            output_name="output",
            s3_output=ProcessingS3Output(
                s3_uri=processing_output_path,
                local_path=local_path,
                s3_upload_mode="EndOfJob"
            )
        ),
    ],
    code="code/fine_tuning.py",
    arguments=[
        "--dataset", dataset_param,
        "--output-path", output_path_param,
        "--mlflow-experiment-name", mlflow_exp_param,
        "--mlflow-run-name", mlflow_run_param,
        "--region", region_param,
        "--role-arn", role_arn_param,
    ],
)

step_process = ProcessingStep(
    name=f"ProcessingSFTTrainer-{unique}",
    step_args=step_args,
)
```

```
pipeline = Pipeline(
    name=f"FineTuningPipeline-{unique}",
    steps=[step_process],
    parameters=[
        dataset_param,
        output_path_param,
        mlflow_exp_param,
        mlflow_run_param,
        region_param,
        role_arn_param,
    ],
    sagemaker_session=pipeline_session
)
```

```
pipeline.upsert(role_arn=role)
```

```
mlflow_exp_name = f"sft-exp-run-{unique}"

execution = pipeline.start(
    parameters={
        "Dataset": dataset_path,
        "OutputPath": output_path,
        "MLflowExperimentName": mlflow_exp_name,
        "MLflowRunName": f"run-{unique}",
        "Region": region,
        "TrainingRoleArn": role,
    }
)
```

```
from rich.pretty import pprint
pprint(pipeline.__dict__)
```

```
execution_arn = execution.arn
%store execution_arn
```

```
%store unique
%store role
%store region
%store prefix
%store s3
```

```
import json

config = {
    "unique": unique,
    "role": role,
    "region": region,
    "prefix": prefix,
    "s3": s3
}

output_file = "context.json"

with open(output_file, "w") as f:
    json.dump(config, f, indent=4)
```

```
pprint(execution.describe())
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

### Inspecting Pipeline Execution Logs

```
%store -r execution_arn
execution_arn
```

```
from sagemaker.mlops.workflow.pipeline import (
    PipelineExecution
)
from sagemaker.core.helper.session_helper import (
    Session
)

sagemaker_session = Session()

execution = PipelineExecution(
    arn=execution_arn,
    sagemaker_session=sagemaker_session,
)
```

```
from rich.pretty import pprint
pprint(execution.describe())
```

```
from pipeline_wrapper import PipelineExecution
pipeline_execution = PipelineExecution(
    execution=execution
)
```

```
pipeline_execution.step(1).name
```

```
pipeline_execution.step(1).logs()
```

```
pipeline_execution.outputs()
```

```
results = pipeline_execution.outputs()[0]

!aws s3 ls {results} --recursive
```

```
!aws s3 cp {results}/results.json results.json
```

```
!cat results.json
```

```
import json
results = json.loads(open("results.json").read())
```

```
%store results
```

## Building and Running the Single-Step Evaluation Pipeline

### Setting Up and Running LLM-as-a-Judge Model Evaluation

```
%store -r results
%store -r unique
%store -r role
%store -r region
%store -r prefix
%store -r s3

print("results:", results)
print("unique:", unique)
print("role:", role)
print("region:", region)
print("prefix:", prefix)
print("s3:", s3)

import json
results = json.loads(open("results.json").read())
```

```
import json

results = json.loads(open("results.json").read())
config = json.loads(open("context.json").read())

unique = config["unique"]
role = config["role"]
region = config["region"]
prefix = config["prefix"]
s3 = config["s3"]
```

```
experiment_name = results['mlflow_experiment_name']
run_name = results['mlflow_run_name'] + "-eval"
```

```
import json
from rich.pretty import pprint

with open("custom_metric.json", "r") as f:
    custom_metric_dict = json.load(f)


pprint(custom_metric_dict)
```

```
custom_metrics = json.dumps([custom_metric_dict])
```

```
eval_ds_path = f"s3://{s3}/{prefix}/eval/gen_qa.jsonl"
eval_output_path = f"s3://{s3}/{prefix}/eval/output"
```

```
!aws s3 cp data/gen_qa.jsonl {eval_ds_path}
```

```
import boto3

sm_client = boto3.client(
    "sagemaker", 
    region_name=region
)
```

```
group_name = results["model_package_group_name"]

response = sm_client.list_model_packages(
    ModelPackageGroupName=group_name,
    SortBy="CreationTime",
    SortOrder="Descending"
)

pprint(response)
```

```
summary_list = response["ModelPackageSummaryList"]
model_package_arn = summary_list[0]['ModelPackageArn']
model_package_arn
```

```
from sagemaker.train.evaluate import (
    LLMAsJudgeEvaluator
)

evaluator_model = 'amazon.nova-pro-v1:0'
builtin_metrics = ["Completeness", "Faithfulness"]
```

```
evaluator = LLMAsJudgeEvaluator(
    model=model_package_arn,
    evaluator_model=evaluator_model,
    dataset=eval_ds_path,
    builtin_metrics=builtin_metrics,
    custom_metrics=custom_metrics,
    s3_output_path=eval_output_path,
    evaluate_base_model=False,
    mlflow_experiment_name=experiment_name,
    mlflow_run_name=run_name
)
```

```
pprint(evaluator)
```

```
execution = evaluator.evaluate()
```

```
pprint(execution)
```

```
execution.wait()
```

```
execution.show_results(
    limit=10, 
    offset=0, 
    show_explanations=False
)
```

```
pprint(execution)
```

### Building and Running the Pipeline

```
%store -r unique
%store -r role
%store -r region
%store -r prefix
%store -r s3

import json
results = json.loads(open("results.json").read())
```

```
import json

results = json.loads(open("results.json").read())
config = json.loads(open("context.json").read())

unique = config["unique"]
role = config["role"]
region = config["region"]
prefix = config["prefix"]
s3 = config["s3"]
```

```
from sagemaker.core.workflow.parameters import (
    ParameterString
)
from sagemaker.core.helper.session_helper import (
    Session, 
    get_execution_role
)
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput, 
    ProcessingOutput, 
    ProcessingS3Output
)
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import (
    ProcessingStep
)
from sagemaker.core.workflow.pipeline_context import (  
    PipelineSession
)
from sagemaker.core.workflow.functions import Join
from sagemaker.core import image_uris
```

```
sagemaker_session = Session()
pipeline_session = PipelineSession()
role = get_execution_role()
```

```
sp = f"s3://{s3}/{prefix}"

results_s3 = f"{sp}/input/"
eval_dataset_s3 = f"{sp}/data/gen_qa.jsonl"
custom_metric_s3 = f"{sp}/input/custom_metric.json"
output_s3 = f"{sp}/output/"

!aws s3 cp data/gen_qa.jsonl {eval_dataset_s3}
!aws s3 cp custom_metric.json {custom_metric_s3}
!aws s3 cp results.json {results_s3}
```

```
results_param = ParameterString(
    name="ResultsPath",
    default_value=results_s3
)

dataset_param = ParameterString(
    name="EvalDataset",
    default_value=eval_dataset_s3
)

metric_param = ParameterString(
    name="CustomMetricPath",
    default_value=custom_metric_s3
)

output_param = ParameterString(
    name="OutputPath",
    default_value=output_s3
)

region_param = ParameterString(
    name="Region",
    default_value=region
)
```

```
eval_processor = ScriptProcessor(
    image_uri=image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    ),
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name=f"llm-eval-{unique}",
    sagemaker_session=pipeline_session,
    role=role,
)
```

```
from sagemaker.core.shapes import (
    ProcessingInput, 
    ProcessingOutput, 
    ProcessingS3Output, 
    ProcessingS3Input
)

from sagemaker.core.workflow.functions import Join
results_s3_uri = Join(
    on="",
    values=[
        results_param,
        "results.json"
    ]
)

results_input = ProcessingInput(
    input_name="results",
    s3_input=ProcessingS3Input(
        s3_uri=results_s3_uri,
        local_path="/opt/ml/processing/input",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="ShardedByS3Key",
    ),
)
```

```
local_path = "/opt/ml/processing/output"

eval_step_args = eval_processor.run(
    inputs=[results_input],
    outputs=[
        ProcessingOutput(
            output_name="output",
            s3_output=ProcessingS3Output(
                s3_uri=output_param,
                local_path=local_path,
                s3_upload_mode="EndOfJob"
            )
        )
    ],
    code="code/evaluation.py",
    arguments=[
        "--eval-dataset", dataset_param,
        "--custom-metric-path", metric_param,
        "--output-path", output_param,
        "--region", region_param,
    ],
)

eval_step = ProcessingStep(
    name=f"LLM-Eval-{unique}",
    step_args=eval_step_args,
)
```

```
pipeline = Pipeline(
    name=f"EvaluationPipeline-{unique}",
    steps=[eval_step],
    parameters=[
        results_param,
        dataset_param,
        metric_param,
        output_param,
        region_param,
    ],
    sagemaker_session=pipeline_session,
)
```

```
pipeline.upsert(role_arn=role)
```

```
execution = pipeline.start()
```

```
eval_pipeline_execution_arn = execution.arn
%store eval_pipeline_execution_arn
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

### Inspecting Pipeline Execution Logs

```
%store -r eval_pipeline_execution_arn
eval_pipeline_execution_arn
```

```
from sagemaker.mlops.workflow.pipeline import (
    PipelineExecution
)
from sagemaker.core.helper.session_helper import (
    Session
)

sagemaker_session = Session()

execution = PipelineExecution(
    arn=eval_pipeline_execution_arn,
    sagemaker_session=sagemaker_session,
)
```

```
from rich.pretty import pprint
pprint(execution.describe())
```

```
from pipeline_wrapper import PipelineExecution

pipeline_execution = PipelineExecution(
    execution=execution
)
```

```
pipeline_execution.step(1).logs()
```

## Configuring and Running a Two-Step Fine-Tuning and Evaluation Pipeline

### Building and Running the Pipeline

```
import random
import string
from pathlib import Path


class UniqueString:
    file_path = Path("unique.txt")

    @classmethod
    def generate_string(cls, length=12):
        return ''.join(
            random.choices(
                string.ascii_lowercase,
                k=length
            )
        )

    @classmethod
    @property
    def value(cls):
        if not hasattr(cls, "_value"):
            if cls.file_path.exists():
                cls._value = cls.file_path.read_text()
                cls._value = cls._value.strip()
            else:
                cls._value = cls.generate_string()
                cls.file_path.write_text(cls._value)

        return cls._value
```

```
print(UniqueString.value)
```

```
from sagemaker.core.workflow.parameters import (
    ParameterString
)
from sagemaker.core.helper.session_helper import (
    Session,
    get_execution_role
)
from sagemaker.core.processing import (
    ScriptProcessor
)
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Output,
    ProcessingS3Input
)
from sagemaker.mlops.workflow.pipeline import (
    Pipeline
)
from sagemaker.mlops.workflow.steps import (
    ProcessingStep
)
from sagemaker.core.workflow.pipeline_context import (
    PipelineSession
)
from sagemaker.core.workflow.functions import (
    Join
)
from sagemaker.core import (
    image_uris
)
```

```
unique = UniqueString.value
```

```
sagemaker_session = Session()
pipeline_session = PipelineSession()
role = get_execution_role()
region = sagemaker_session.boto_region_name

prefix = f"pipeline-{unique}"
s3 = sagemaker_session.default_bucket()
```

```
train_dataset_s3 = (
    f"s3://{s3}/{prefix}/data/data.jsonl"
)

eval_dataset_s3 = (
    f"s3://{s3}/{prefix}/data/gen_qa.jsonl"
)

custom_metric_s3 = (
    f"s3://{s3}/{prefix}/input/custom_metric.json"
)

train_output_s3 = (
    f"s3://{s3}/{prefix}/training-output/"
)

eval_output_s3 = (
    f"s3://{s3}/{prefix}/evaluation-output/"
)
```

```
!aws s3 cp data/data.jsonl {train_dataset_s3}
!aws s3 cp data/gen_qa.jsonl {eval_dataset_s3}
!aws s3 cp custom_metric.json {custom_metric_s3}
```

```
role_arn_param = ParameterString(
    name="TrainingRoleArn",
    default_value=role
)

train_dataset_param = ParameterString(
    name="Dataset",
    default_value=train_dataset_s3
)

train_output_param = ParameterString(
    name="TrainingOutputPath",
    default_value=train_output_s3
)
```

```
mlflow_exp_param = ParameterString(
    name="MLflowExperimentName",
    default_value=f"sft-experiment-{unique}"
)

mlflow_run_param = ParameterString(
    name="MLflowRunName",
    default_value=f"run-{unique}"
)

region_param = ParameterString(
    name="Region",
    default_value=region
)
```

```
eval_dataset_param = ParameterString(
    name="EvalDataset",
    default_value=eval_dataset_s3
)

custom_metric_param = ParameterString(
    name="CustomMetricPath",
    default_value=custom_metric_s3
)

eval_output_param = ParameterString(
    name="EvaluationOutputPath",
    default_value=eval_output_s3
)
```

```
local_path = "/opt/ml/processing/output"

train_processor = ScriptProcessor(
    image_uri=image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    ),
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name=f"sfttrainer-job-{unique}",
    sagemaker_session=pipeline_session,
    role=role,
)
```

```
from sagemaker.mlops.workflow.steps import CacheConfig

cache_config = CacheConfig(
    enable_caching=True, 
    expire_after="PT12H"
)
```

```
train_step_args = train_processor.run(
    inputs=[],
    outputs=[
        ProcessingOutput(
            output_name="output",
            s3_output=ProcessingS3Output(
                s3_uri=train_output_s3,
                local_path=local_path,
                s3_upload_mode="EndOfJob"
            )
        )
    ],
    code="code/fine_tuning.py",
    arguments=[
        "--dataset", train_dataset_param,
        "--output-path", train_output_param,
        "--mlflow-experiment-name", mlflow_exp_param,
        "--mlflow-run-name", mlflow_run_param,
        "--region", region_param,
        "--role-arn", role_arn_param,
    ],
)

train_step = ProcessingStep(
    name=f"FineTuning-{unique}",
    step_args=train_step_args,
    cache_config=cache_config
)
```

```
eval_processor = ScriptProcessor(
    image_uri=image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    ),
    instance_type="ml.m5.xlarge",
    instance_count=1,
    base_job_name=f"llm-eval-{unique}",
    sagemaker_session=pipeline_session,
    role=role,
)
```

```
tpoc = train_step.properties.ProcessingOutputConfig
ts3uri = tpoc.Outputs["output"].S3Output.S3Uri

results_input = ProcessingInput(
    input_name="results",
    s3_input=ProcessingS3Input(
        s3_uri=Join(
            on="",
            values=[ts3uri, "results.json"]
        ),
        local_path="/opt/ml/processing/input",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
    ),
)
```

```
local_path = "/opt/ml/processing/output"

eval_step_args = eval_processor.run(
    inputs=[
        results_input
    ],
    outputs=[
        ProcessingOutput(
            output_name="output",
            s3_output=ProcessingS3Output(
                s3_uri=eval_output_param,
                local_path=local_path,
                s3_upload_mode="EndOfJob"
            )
        )
    ],
    code="code/evaluation.py",
    arguments=[
        "--eval-dataset", eval_dataset_param,
        "--custom-metric-path", custom_metric_param,
        "--output-path", eval_output_param,
        "--region", region_param,
    ],
)

eval_step = ProcessingStep(
    name=f"Evaluation-{unique}",
    step_args=eval_step_args,
    depends_on=[train_step],
    cache_config=cache_config
)
```

```
pipeline = Pipeline(
    name=f"CompletePipeline-{unique}",
    steps=[
        train_step,
        eval_step
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
    }
)
```

```
pipeline_execution_arn = execution.arn
%store pipeline_execution_arn
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

### Inspecting Pipeline Execution Logs 

```
%store -r pipeline_execution_arn
pipeline_execution_arn
```

```
from sagemaker.mlops.workflow.pipeline import (
    PipelineExecution
)
from sagemaker.core.helper.session_helper import (
    Session
)

sagemaker_session = Session()

execution = PipelineExecution(
    arn=pipeline_execution_arn,
    sagemaker_session=sagemaker_session,
)
```

```
from rich.pretty import pprint
pprint(execution.describe())
```

```
from pipeline_wrapper import PipelineExecution

pipeline_execution = PipelineExecution(
    execution=execution
)
```

```
pipeline_execution.number_of_steps()
```

```
pipeline_execution.step(1).name
```

```
pipeline_execution.step(1).logs()
```

```
pipeline_execution.step(2).name
```

```
pipeline_execution.step(2).logs()
```

```
pipeline_execution.outputs()
```

```
output = pipeline_execution.outputs()[0]
s3_path = output + "training_results.json"
```

```
!aws s3 ls {s3_path}
```

```
print(s3_path)
```

## Preparing the Lambda functions for deploying a model to an endpoint

### Setting Up a Lambda Function that reads a JSON file in S3

```
{
  "s3_path": "<S3 PATH>"
}
```

### Setting Up the Lambda Function for checking if an endpoint exists

```
{
  "endpoint_name": "endpoint-100"
}
```

### Setting Up the Lambda Function for deploying a model to a new endpoint

```
{
  "endpoint_name": "endpoint-100",
  "model_arn": "<MODEL ARN>",
  "instance_type": "ml.g5.4xlarge"
}
```

### Setting Up the Lambda Function for deploying a model to an existing endpoint

```
{
  "endpoint_name": "endpoint-100",
  "model_arn": "<MODEL ARN>",
  "instance_type": "ml.g5.4xlarge"
}
```

## Completing the LLMOps pipeline

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

## Links and Resources

Here are the links and resources referenced throughout this chapter for your convenience:

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
    <th align="center" width="200px">◀ PREVIOUS</th>
    <th align="center" width="200px">HOME</th>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter08">
        CHAPTER 08
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md">
        DETAILS.md
      </a>
    </td>
  </tr>
</table>
