# Practical Data Management on AWS

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

## Running SQL queries in Amazon Athena

### Using the AWS CLI to run SQL queries with Amazon Athena
### Using Boto3 to run SQL queries with Amazon Athena

## Ingesting data into a SageMaker Feature Store

### Preparing the data to be ingested into the Feature Store
### Ingesting data into a Feature Store

## Adding searchable metadata to the features

### Updating the feature metadata
### Searching features in an existing feature group using the metadata

## Retrieving data from the online and offline feature stores
### Retrieving data from the online feature store
### Retrieving data from the offline feature store
