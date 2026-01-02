<img width="468" height="22" alt="image" src="https://github.com/user-attachments/assets/1fa6976c-71e1-42bb-b29a-31f3a0379a41" /><img width="468" height="128" alt="image" src="https://github.com/user-attachments/assets/3dbd037a-5b57-4b04-905c-cb6ab658adc6" />## Technical requirements

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

## Preparing and processing the synthetic data

```
pip install ipython pandas fastparquet
```

```
ipython
```

```
import pandas as pd

url_directory = ''.join([
    "https://raw.githubusercontent.com/",
    "PacktPublishing/",
    "Machine-Learning-Engineering-on-AWS",
    "-Second-Edition/",
    "refs/heads/main/chapter05/"
])
```

```
def get_url(filename, url_directory=url_directory):
    return url_directory + filename
```

```
data_up_to_2024 = get_url("data_up_to_2024.csv")
df_up_to_2024 = pd.read_csv(data_up_to_2024)

data_2025_onwards = get_url("data_2025_onwards.csv")
df_2025_onwards = pd.read_csv(data_2025_onwards)
```

```
df_up_to_2024.head(20)
```

```
df_2025_onwards.head(20)
```

```
import uuid

def generate_id():
    return str(uuid.uuid4())
```

```
df_up_to_2024['unique_id'] = df_up_to_2024.apply(
    lambda _: generate_id(), axis=1
)

df_2025_onwards['unique_id'] = df_2025_onwards.apply(
    lambda _: generate_id(), axis=1
)
```

```
unique_column = df_up_to_2024.pop('unique_id') 
df_up_to_2024.insert(0, 'unique_id', unique_column)

unique_column = df_2025_onwards.pop('unique_id') 
df_2025_onwards.insert(0, 'unique_id', unique_column)
```

```
df_up_to_2024.head(10), df_2025_onwards.head(10)
```

```
df_up_to_2024.to_csv(
    'data_up_to_2024.csv', 
    index=False
)
```

```
df_2025_onwards.to_parquet(
    "data_2025_onwards.parquet", 
    engine="fastparquet", 
    index=False
)
```

```
exit
```

## Inspecting the CSV and Parquet files

```
head -n 20 data_up_to_2024.csv
```

```
pip install parquet-tools
```

```
parquet-tools show data_2025_onwards.parquet
```

```
parquet-tools inspect data_2025_onwards.parquet
```

## Uploading the CSV and Parquet files to an S3 bucket

```
S3_BUCKET_NAME="<S3 BUCKET NAME>"
aws s3 mb s3://$S3_BUCKET_NAME
```

```
CSV_FILE="data_up_to_2024.csv"
PARQUET_FILE="data_2025_onwards.parquet"
aws s3 cp $CSV_FILE s3://$S3_BUCKET_NAME/
aws s3 cp $PARQUET_FILE s3://$S3_BUCKET_NAME/
```

## Launching up an Amazon EMR cluster with Apache Iceberg installed

### Setting up an Amazon EMR cluster with Apache Iceberg installed

```
[
  {
    "Classification":"iceberg-defaults",
    "Properties":{"iceberg.enabled":"true"}
  }
]
```

## Performing Apache Iceberg queries on S3 tables with Apache Spark

### Accessing the Spark shell

```
ACCOUNT_ID=`aws sts get-caller-identity --query "Account" --output text`
```

```
TABLE_BUCKET="<SPECIFY S3 TABLE BUCKET NAME>"
```

```
PACKAGES=software.amazon.s3tables:s3-tables-catalog-for-iceberg-runtime:0.1.3
CONF1=spark.sql.catalog.s3tablesbucket=org.apache.iceberg.spark.SparkCatalog
CONF2=spark.sql.catalog.s3tablesbucket.catalog-impl=software.amazon.s3tables.iceberg.S3TablesCatalog
CONF3=spark.sql.catalog.s3tablesbucket.warehouse=arn:aws:s3tables:us-east-1:$ACCOUNT_ID:bucket/$TABLE_BUCKET
CONF4=spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
```

```
pyspark \
--packages $PACKAGES \
--conf $CONF1 \
--conf $CONF2 \
--conf $CONF3 \
--conf $CONF4 \
--conf spark.ui.showConsoleProgress=false \
--conf spark.log.level=ERROR
```

### Creating a namespace

```
def SQL(statement):
    spark.sql(statement).explain(True)
    spark.sql(statement).show(40, truncate=False)
```

```
namespace = "sentiments_namespace"
```

```
SQL(f"""
CREATE NAMESPACE IF NOT EXISTS s3tablesbucket.{namespace}
""")
```

```
SQL("SHOW NAMESPACES IN s3tablesbucket")
```

```
s3_bucket = "<S3 BUCKET NAME>"
```

```
df_up_to_2024 = spark.read.csv(
    f"s3://{s3_bucket}/data_up_to_2024.csv", 
    header=True, 
    inferSchema=True
)
```

```
df_up_to_2024.show(20)
```

```
name="sentiments_v1"
table_name = f"s3tablesbucket.{namespace}.{name}"
```

```
SQL(f"""
CREATE OR REPLACE TABLE {table_name}(
  unique_id STRING,
  statement STRING,
  score DOUBLE,
  tag STRING,
  is_spam BOOLEAN,
  date DATE
)
USING iceberg
TBLPROPERTIES ('format-version'='2')
""")
```

```
df_up_to_2024.writeTo(table_name) \
    .using("iceberg") \
    .tableProperty("format-version","2") \
    .append()
```

```
SQL(f"SELECT * FROM { table_name }")
```

```
SQL(f"SELECT COUNT(*) FROM { table_name }")
```

```
SQL(f"""
SELECT COUNT(*) FROM { table_name } WHERE is_spam=true
""")
```

### Loading data from a Parquet file to a table

```
df_2025_onwards = spark.read.parquet(
    f"s3://{s3_bucket}/data_2025_onwards.parquet" 
)
```

```
df_2025_onwards.show(20)
```

```
from pyspark.sql.functions import to_date

df_2025_onwards \
    .withColumn("date", 
                to_date("date", "yyyy-MM-dd")) \
    .writeTo(table_name) \
    .using("iceberg") \
    .tableProperty("format-version","2") \
    .append()
```

```
SQL(f"SELECT COUNT(*) FROM { table_name }")
```

```
SQL(f"""
SELECT * FROM { table_name } WHERE year(date) >= 2025
""")
```

### Performing time travel queries on S3 tables

```
SQL(f"SELECT * FROM {table_name}.history LIMIT 10")
```

```
snapshot_id = "<SNAPSHOT ID>"
```

```
SQL(f"""
SELECT COUNT(*) FROM {table_name}
for system_version as of {snapshot_id}
""")
```

```
SQL(f"""
SELECT * FROM {table_name} 
FOR system_version AS OF {snapshot_id} 
ORDER BY date DESC
LIMIT 30
""")
```

```
spark.sql(f"ALTER TABLE {table_name} DROP COLUMN score")
```

```
SQL(f"SELECT * FROM {table_name} LIMIT 30")
```

```
SQL(f"""
SELECT * FROM {table_name} 
FOR system_version AS OF {snapshot_id} 
ORDER BY date DESC
LIMIT 30
""")
```

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
!aws s3 mb {output_bucket}
```

```
import boto3
region_name = 'us-east-1'
athena = boto3.client('athena', 
                      region_name=region_name)
```

```
table_bucket = '<TABLE BUCKET NAME>
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
df = SQL(f"SELECT * FROM { table }")
print(df)
df.to_csv("tmp/sentiments_data.csv", index=False)
```
