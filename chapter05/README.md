# Chapter 5: Practical Data Management on AWS

In this chapter, you'll explore AWS services and capabilities that support data management for analytics and machine learning workloads. You will work with AWS Lake Formation permissions, query data using Amazon Athena, ingest data into Amazon SageMaker Feature Store, and retrieve data from both the online and offline feature stores.

To help you build practical data management skills for modern cloud-based ML workflows, we will cover the following topics in this chapter:

- Working with AWS Lake Formation permissions
- Running SQL queries in Amazon Athena
- Ingesting data into a SageMaker feature store
- Adding searchable metadata to the features
- Retrieving data from the online and offline feature stores

This README.md file contains the commands and code snippets referenced in a chapter of *Machine Learning Engineering on AWS — Second Edition* by Joshua Arvin Lat, published by Packt. It is intended to support the examples in the book by making it simpler for you to copy, run, and modify the provided materials.

![Machine Learning Engineering on AWS 2nd ed](../books.png)

To help you get started more easily, the repository includes a [DETAILS.md](https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md) file containing additional guidance, references, and important notes for the examples discussed throughout the book.

## Technical requirements

We must have the following ready before we jump into the hands-on examples of this chapter:

- **A new IAM role named SageMakerAdminRole with the AmazonSageMakerAccessFullAccess and AmazonS3FullAccess permission policies attached**: You can create this IAM role by first opening the AWS Management Console, typing IAM in the search bar, and then selecting IAM from the list of results. Once in the IAM console, select Roles in the left-hand menu and then click the Create role button. From there, select Custom trust policy from the list of options under Trusted entity type. Replace the custom trust policy with the following trust policy that allows SageMaker to assume the role:

    ```
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    ```

    Continue to the permissions page and attach the AmazonSageMakerFullAccess and AmazonS3FullAccess policies. After completing the steps and saving the role (using the name SageMakerAdminRole), it will be ready for use by SageMaker resources that require these permissions. This role essentially grants SageMaker the administrative-level permissions required to operate fully within your AWS account.

- **A new IAM user with AdministratorAccess permissions**: You can create this IAM user through the IAM Console, by typing IAM in the search bar, and then selecting IAM from the list of results. Once in the IAM console, select Users from the left-hand menu and then click the Create user button. Enter the desired user name (superadmin), ensure that the checkbox for Provide user access to the AWS Management Console is checked, choose I want to create an IAM user, and click Next. On the Set permissions page, choose Attach policies directly, then search for and select the AdministratorAccess policy. Continue through the remaining steps, and finally clicking the Create user button. After completion, you'll be able to use the new IAM user to log in to the AWS Management Console with full administrative permissions.

| Note |
|:-----|
| An IAM user with administrative-level permissions (for example, one attached with the AdministratorAccess policy) can perform most actions across AWS services required in this chapter. While it is generally recommended that IAM users follow the principle of least privilege and have more restrictive permissions, attaching the AdministratorAccess policy to your IAM user will simplify setup and should allow you to complete the hands-on examples in this chapter without permissions-related issues. |

Before proceeding, confirm that your currently selected AWS Region matches the region where your VPC has been set up. You can change the region at any time by using the region selector in the upper-right corner of the AWS Management Console and choosing the correct region (for example, us-east-1) from the dropdown list.


## Running SQL queries in Amazon Athena

### Using the AWS CLI to run SQL queries with Amazon Athena

```
ATHENA_OUTPUT_LOCATION="<NAME OF NEW S3 BUCKET>"
aws s3 mb s3://$ATHENA_OUTPUT_LOCATION
```

```
TABLE_BUCKET_NAME="<TABLE BUCKET NAME>"
```

```
CATALOG_BUCKET="s3tablescatalog/$TABLE_BUCKET_NAME"
NAMESPACE="sentiments_namespace"
TABLE="sentiments_v1"
QUERY_STRING="SELECT * FROM \"$CATALOG_BUCKET\".$NAMESPACE.$TABLE LIMIT 10"
echo $QUERY_STRING
```

```
OUTPUT_LOC="$ATHENA_OUTPUT_LOCATION"
RESULT_CONF="OutputLocation=s3://$OUTPUT_LOC/"

aws athena start-query-execution \
  --query-string "$QUERY_STRING" \
  --work-group "primary" \
  --result-configuration $RESULT_CONF
```

```
aws athena get-query-execution \
  --query-execution-id <QUERY_EXECUTION_ID>
```

```
aws s3 cp <SOURCE S3 PATH> output.csv
```

```
cat output.csv
```

### Using Boto3 to run SQL queries with Amazon Athena

```
pip install ipython pandas fsspec s3fs
```

```
ipython
```

```
athena_output_s3_bucket_name = "<BUCKET NAME>"
output_bucket = "s3://" + athena_output_s3_bucket_name
```

```
import boto3
region_name = 'us-east-1'
athena = boto3.client('athena', 
                      region_name=region_name)
```

```
table_bucket = '<TABLE BUCKET NAME>'
```

```
t_a = f'"s3tablescatalog/{table_bucket}"'
t_b = 'sentiments_namespace'
t_c = 'sentiments_v1'
table = f'{t_a}.{t_b}.{t_c}'
query = f"SELECT * FROM { table }"
database = f"s3tablescatalog/{table_bucket}"

query_config = {
    'database': database,
    'output_bucket': output_bucket
}
```

```
def execute_query(query, query_config=query_config):
    location = query_config['output_bucket']
    query_params = {
        'QueryString': query,
        'QueryExecutionContext': { 
            'Database': query_config['database']
        },
        'ResultConfiguration': {
            'OutputLocation': location
        }
    }   
    response = athena.start_query_execution(
        **query_params
    )
    print(response)
    return response['QueryExecutionId']
```

```
def get_s3_output_path(athena_execution_id):
   details = athena.get_query_execution(
      QueryExecutionId = athena_execution_id
   )
   print(details)
   execution = details['QueryExecution']
   configuration = execution['ResultConfiguration']
   return configuration['OutputLocation']
```

```
athena_execution_id = execute_query(query)
output_path = get_s3_output_path(athena_execution_id)
output_path
```

```
!aws s3 cp {output_path} tmp/output.csv
```

```
import pandas as pd
pd.read_csv("tmp/output.csv")
```

```
from time import sleep

def SQL(statement):
    print(statement)
    execution_id = execute_query(statement)
    print('Sleeping for 15 seconds')
    sleep(15)

    if 'DELETE' in statement:
        return None
    else:
        output_path = get_s3_output_path(execution_id)
        return pd.read_csv(output_path)
```

```
SQL(f"SELECT * FROM { table } WHERE is_spam = True")
```

```
SQL(f"DELETE FROM { table } WHERE is_spam = True")
```

```
SQL(f"""SELECT COUNT(*) AS spam_count FROM { table } WHERE is_spam = True""")
```

```
df = SQL(f"SELECT * FROM { table }")
print(df)
```

```
df.to_csv("tmp/sentiments_data.csv", index=False)
```

```
exit
```

```
UPLOAD_BUCKET="<NAME OF NEW S3 BUCKET>"
aws s3 mb s3://$UPLOAD_BUCKET
```

```
aws s3 cp tmp/sentiments_data.csv s3://$UPLOAD_BUCKET/
```

## Ingesting data into a SageMaker Feature Store

### Setting up a SageMaker Notebook instance

```
UPLOAD_BUCKET="<NAME OF EXISTING S3 BUCKET>"
aws s3 cp s3://$UPLOAD_BUCKET/sentiments_data.csv .
```

```
cp sentiments_data.csv SageMaker/
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
from sagemaker.mlops.feature_store import (
    FeatureGroup,
    FeatureMetadata,
    OnlineStoreConfig,
    OfflineStoreConfig,
    S3StorageConfig,
    load_feature_definitions_from_dataframe,
    ingest_dataframe,
    create_athena_query,
)
```

### Preparing the data to be ingested into the Feature Store

```
import pandas as pd
df = pd.read_csv("sentiments_data.csv")
```

```
df = df.drop('is_spam', axis=1)
```

```
from datetime import datetime

def get_event_time(date_string):
    obj = datetime.strptime(date_string, '%Y-%m-%d')
    output = float(obj.timestamp())
    return output
```

```
first_event_time = get_event_time('2024-01-01')
df['event_time'] = first_event_time
```

```
df.to_csv("feature_store_01.csv", index=False)
```

```
second_event_time = get_event_time('2025-01-01')

for index, row in df.iterrows():
    if row['tag'] == 'NEUTRAL':
        new_tag = 'POSITIVE'        
        df.at[index, 'tag'] = new_tag
        df.at[index, 'event_time'] = second_event_time
```

```
df.to_csv("feature_store_02.csv", index=False)
```

### Ingesting data into a Feature Store

```
from time import sleep
from boto3 import Session as BotoSession

boto_session = BotoSession()
region = boto_session.region_name
client = boto_session.client

c1 = client(service_name='sagemaker', 
            region_name=region)

service_name='sagemaker-featurestore-runtime'
c2 = client(service_name=service_name,
            region_name=region)
```

```
import pandas as pd
df_01 = pd.read_csv("feature_store_01.csv")
df_02 = pd.read_csv("feature_store_02.csv")
```

```
defs = load_feature_definitions_from_dataframe(df_01)
```

```
import boto3
sts = boto3.client("sts")
account_id = sts.get_caller_identity()["Account"]

s3_bucket_name = f"sagemaker-{region}-{account_id}"
s3_input = f"s3://{s3_bucket_name}/store/input"
s3_output = f"s3://{s3_bucket_name}/store/output"
```

```
from sagemaker.core.helper.session_helper import (
    get_execution_role
)

role_arn = get_execution_role()
```

```
group_name = "sentiments-feature-group"

FeatureGroup.create(
    feature_group_name=group_name,
    feature_definitions=defs,
    record_identifier_feature_name="unique_id",
    event_time_feature_name="event_time",
    role_arn=role_arn,
    online_store_config=OnlineStoreConfig(
        enable_online_store=True
    ),
    offline_store_config=OfflineStoreConfig(
        s3_storage_config=S3StorageConfig(
            s3_uri=s3_input
        )
    ),
)
```

```
def get_status(group_name="sentiments-feature-group"):
    fg = FeatureGroup.get(
        feature_group_name=group_name
    )
    return fg.feature_group_status

print(get_status())
```

```
while get_status() == "Creating":
    print("Waiting...")
    sleep(10)

print(get_status())
```

```
ingest_dataframe(
    feature_group_name="sentiments-feature-group",
    data_frame=df_01,
    max_workers=3,
    wait=True,
)

ingest_dataframe(
    feature_group_name="sentiments-feature-group",
    data_frame=df_02,
    max_workers=3,
    wait=True,
)
```

## Adding searchable metadata to the features

```
metadata = FeatureMetadata.get(
    feature_group_name="sentiments-feature-group",
    feature_name="unique_id"
)
metadata.update(
    description="UUID of the record"
)
```

```
from sagemaker.mlops.feature_store import (
    FeatureParameter
)

metadata.update(
    description="UUID of the record",
    parameter_additions=[
        FeatureParameter(
            key="generated", 
            value="true"
        ),
        FeatureParameter(
            key="another-key", 
            value="another-value"
        ),
    ]
)
```

```
descriptions = {
    "unique_id": "UUID of the record",
    "statement": "Statement or comment of a user",
    "tag": "Sentiment tag based on score",
    "date": "When the comment was shared",
    "event_time": "Feature store event time value",
}

for feature_name, desc in descriptions.items():
    meta = FeatureMetadata.get(
        feature_group_name="sentiments-feature-group",
        feature_name=feature_name
    )
    meta.update(description=desc)
```

```
c1.search(Resource="FeatureMetadata")
```

```
c1.search(
    Resource="FeatureMetadata", 
    SearchExpression={'Filters': [
        {'Name': 'FeatureType', 
         'Operator': 'Equals', 
         'Value': 'Fractional'}
    ]}
)
```

```
c1.search(
    Resource="FeatureMetadata", 
    SearchExpression={'Filters': [
        {'Name': 'Description', 
         'Operator': 'Contains', 
         'Value': 'comment'}
    ]}
)
```

```
c1.search(
    Resource="FeatureMetadata", 
    SearchExpression={'Filters': [
        {'Name': 'Parameters.generated', 
         'Operator': 'Equals', 
         'Value': 'true'}
    ]}
)
```

## Retrieving data from the online and offline feature stores

```
first_id = df_01.iloc[0].unique_id
```

```
first_id
```

```
feature_group = FeatureGroup.get(
    feature_group_name="sentiments-feature-group"
)

response = feature_group.get_record(
    record_identifier_value_as_string=str(first_id)
)
```

```
from pprint import pprint

clean = {
    "expires_at": response.expires_at,
    "record": [
        { "feature": f.feature_name, 
          "value": f.value_as_string }
        for f in response.record
    ]
}

pprint(
    clean, 
    indent=2, 
    width=60, 
    compact=True, 
    sort_dicts=False
)
```

```
offline_config = feature_group.offline_store_config
storage_config = offline_config.s3_storage_config
config_s3_uri = storage_config.resolved_output_s3_uri
!aws s3 ls { config_s3_uri } --recursive
```

```
from sagemaker.core.helper.session_helper import (
    Session
)

session = Session()
group_name = "sentiments-feature-group"

query = create_athena_query(
    group_name, 
    session=session
)
```

```
pprint(vars(query))
```

```
table = query.table_name
```

```
def SQL(
    statement: str,
    feature_group_name="sentiments-feature-group",
    session=session,
    output_location=s3_output,
    no_df=False
):
    print("---" * 20)
    print(f"QUERY: {statement}")
    print("---" * 20)

    query = create_athena_query(
        feature_group_name,
        session=session
    )

    query.run(
        query_string=statement,
        output_location=output_location
    )

    query.wait()

    if not no_df:
        return query.as_dataframe()

    return None
```

```
SQL(f"SELECT * FROM { table }")
```

```
SQL(f"SELECT COUNT(*) FROM { table }")
```

```
SQL(f"""
WITH latest_versions AS (
    SELECT *,ROW_NUMBER() 
    OVER (
        PARTITION BY unique_id 
        ORDER BY event_time DESC
   ) AS rn
    FROM {table}
)
SELECT * FROM latest_versions WHERE rn = 1;
""")
```

```
SQL(f"""
CREATE OR REPLACE VIEW latest_versions_view AS
WITH latest_versions AS (
    SELECT *,ROW_NUMBER() 
       OVER (
           PARTITION BY unique_id 
           ORDER BY event_time DESC
       ) AS rn
    FROM { table }
)
SELECT * FROM latest_versions WHERE rn = 1;
""", no_df=True)
```

```
SQL("SELECT * FROM latest_versions_view")
```

```
feature_group.delete()
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
