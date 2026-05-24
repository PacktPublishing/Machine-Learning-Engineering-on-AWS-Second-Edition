# Chapter 6: Pragmatic Data Processing on AWS

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
