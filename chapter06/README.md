# Chapter 6: Pragmatic Data Processing on AWS

In this chapter, you'll learn how to use SageMaker Processing jobs for resource-intensive data processing workloads. You will run a back-translation workflow using SageMaker Processing, prepare datasets and scripts, and explore best practices for designing, managing, scaling, and securing data processing workflows.

We will cover the following topics in this chapter:

- Getting started with SageMaker Processing jobs
- Running your first SageMaker Processing job
- Preparing the input data and script for the back translation job
- Automating back translation workflows with SageMaker Processing jobs

This README.md file contains the commands and code snippets referenced in a chapter of *Machine Learning Engineering on AWS — Second Edition* by Joshua Arvin Lat, published by Packt. It is intended to support the examples in the book by making it simpler for you to copy, run, and modify the provided materials.

![Machine Learning Engineering on AWS 2nd ed](../books.png)

To help you get started more easily, the repository includes a [DETAILS.md](https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md) file containing additional guidance, references, and important notes for the examples discussed throughout the book.

## Technical requirements

Before we start, we must have the following ready:

- The SageMaker notebook instance set up from Chapter 5, Practical Data Management on AWS, configured with the required IAM role (SageMakerAdminRole). Make sure to start the notebook instance before working on the hands-on portion of this chapter.
  
- Applied account-level quota value under ml.p3.2xlarge for processing job usage to be 2 (or above). You can review and request quota increases through the Service Quotas console (within the AWS Management Console). To view the applied quota values and perform the required quota increase request for the ml.p3.2xlarge for processing job usage applied account-level quota value, simply navigate to the link https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas and locate the quota value to be increased.
  
- Applied account-level quota value under ml.m5.xlarge for processing job usage to be 2 (or above).

| Note |
|:-----|
| When requesting or verifying the quota, make sure that you select "ml.m5.xlarge for processing job usage" (or "ml.p3.2xlarge for processing job usage" if you plan to use GPU-based instances). SageMaker provides several instance-level quota categories, such as cluster usage, cluster spot instance usage, endpoint usage, notebook instance usage, processing job usage, spot training usage, and training job usage. Since this chapter focuses on SageMaker Processing jobs, ensure that you modify the quota specifically for the processing job usage category for the appropriate instance type rather than the other categories. |


## Running your first SageMaker Processing job

```
pwd
```

```
mkdir -p scripts/000
cd scripts/000
```

```
touch processing.py
```

```
if __name__=='__main__':
    print('Running!')
```

```
python processing.py
```

```
%pip uninstall -y sagemaker
%pip install "sagemaker==3.5.0"
```

```
import warnings

m = "Field .* has conflict with protected namespace"

warnings.filterwarnings(
    "ignore",
    message=m
)
```

```
from sagemaker.core.helper.session_helper import (
    Session, 
    get_execution_role
)

session = Session()
execution_role = get_execution_role()
region = session.boto_region_name
```

```
instance_type = "ml.m5.xlarge"
instance_count = 1
```

```
from sagemaker.core.image_uris import (
    get_training_image_uri
)
from sagemaker.core.processing import (
    FrameworkProcessor
)

image_uri = get_training_image_uri(
    region=region,
    framework="pytorch",
    framework_version="1.13",
    py_version="py39",
    instance_type=instance_type,
)

processor = FrameworkProcessor(
    image_uri=image_uri,
    role=execution_role,
    instance_type=instance_type,
    instance_count=instance_count,
)
```

```
import random
import string

def random_string(length=6):
    return ''.join(random.choices(
        string.ascii_uppercase, 
        k=length
    ))
```

```
job_name = f"processing-job-{random_string()}"
job_name
```

```
import os

source_dir = os.path.abspath("scripts/000")
code = "processing.py"
```

```
processor.run(
    code=code,
    source_dir=source_dir,
    job_name=job_name,
    wait=True,
)
```

## Preparing the input data and script for the back translation job

```
pwd
```

```
mkdir -p input output scripts/001
```

```
DIR_PATH=https://raw.githubusercontent.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/refs/heads/main/chapter06
CSV_PATH=$DIR_PATH/positive_sentiments_tagalog_250.csv

wget $CSV_PATH -O input/input_tagalog_250.csv
```

```
head -n 6 input/input_tagalog_250.csv
```

```
ORIG_FILE="input/input_tagalog_250.csv"
NEW_FILE="input/input.csv"
head -n 10 $ORIG_FILE > $NEW_FILE
```

```
REQ_PATH=$DIR_PATH/scripts/001/requirements.txt
wget $REQ_PATH -O scripts/001/requirements.txt
```

```
cat scripts/001/requirements.txt
```

```
SCRIPT_PATH=$DIR_PATH/scripts/001/processing.py
wget $SCRIPT_PATH -O scripts/001/processing.py
```

```
less -N scripts/001/processing.py
```

## Automating back-translation workflows with SageMaker Processing jobs

```
%pip uninstall -y sagemaker
%pip install "sagemaker==3.5.0"
```

```
import warnings

m = "Field .* has conflict with protected namespace"

warnings.filterwarnings(
    "ignore",
    message=m
)
```

```
from sagemaker.core.helper.session_helper import (
    Session, 
    get_execution_role
)

session = Session()
execution_role = get_execution_role()
region = session.boto_region_name
```

```
instance_type = "ml.p3.2xlarge"
instance_count = 1
```

```
from sagemaker.core.image_uris import (
    get_training_image_uri
)
from sagemaker.core.processing import (
    FrameworkProcessor
)

image_uri = get_training_image_uri(
    region=region,
    framework="pytorch",
    framework_version="2.0",
    py_version="py310",
    instance_type=instance_type,
)

processor = FrameworkProcessor(
    image_uri=image_uri,
    role=execution_role,
    instance_type=instance_type,
    instance_count=instance_count,
)
```

```
import random
import string

def random_string(length=6):
    return ''.join(random.choices(
        string.ascii_uppercase, 
        k=length
    ))
```

```
random_id = random_string()
job_name = f"processing-job-{ random_id }"
job_name
```

```
import os

source_dir = os.path.abspath("scripts/001")
code = "processing.py"
```

```
from sagemaker.core.shapes import (
    ProcessingInput, 
    ProcessingS3Input,
    ProcessingOutput, 
    ProcessingS3Output
)
```

```
s3_data_path = ('').join([
    "s3://",
    session.default_bucket(),
    "/" + random_id,
])

input_data_path = s3_data_path + "/input"
output_data_path = s3_data_path + "/output"
```

```
print(s3_data_path)
print(input_data_path)
print(output_data_path)
```

```
!aws s3 cp input/input.csv {input_data_path}/input.csv
```

```
!aws s3 ls {input_data_path} --recursive
```

```
languages = str(['en','pt','de'])

arguments = [
    "--source_language", "tl", 
    "--languages_for_back_translation", languages,
]
```

```
processing_s3_input = ProcessingS3Input(
    s3_uri=input_data_path,
    local_path="/opt/ml/processing/input",
    s3_data_type="S3Prefix",
    s3_input_mode="File",
    s3_data_distribution_type="ShardedByS3Key",
)

inputs = [
    ProcessingInput(
        input_name="input-00",
        s3_input=processing_s3_input
    )
]
```

```
processing_s3_output = ProcessingS3Output(
    s3_uri=output_data_path,
    local_path="/opt/ml/processing/output",
    s3_upload_mode="EndOfJob",
)

outputs = [
    ProcessingOutput(
            output_name="output-00",
            s3_output=processing_s3_output
        ),
]
```

```
processor.run(
    code=code,
    source_dir=source_dir,
    job_name=job_name,
    arguments=arguments,
    inputs=inputs,
    outputs=outputs,
    wait=True,
)
```

```
job = processor.jobs[0]
output_config = job.processing_output_config
s3_output = output_config.outputs[0].s3_output
s3_output
```

```
!aws s3 ls {s3_output.s3_uri} --recursive
```

```
dl_path = s3_output.s3_uri + "/output.csv"
```

```
!aws s3 cp {dl_path} output/output.csv
```

```
!cat output/output.csv
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
    <th align="center" width="200px">◀ PREVIOUS</th>
    <th align="center" width="200px">HOME</th>
    <th align="center" width="200px">NEXT ▶</th>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter05">
        CHAPTER 05
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md">
        DETAILS.md
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/tree/main/chapter07">
        CHAPTER 07
      </a>
    </td>
  </tr>
</table>
