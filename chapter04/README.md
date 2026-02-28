# Chapter 3: Modernizing Analytics with a Managed Transactional Data Lake

## Preparing and processing the synthetic data

### Downloading and preparing the synthetic tabular data

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

### Inspecting the CSV and Parquet files

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

### Uploading the CSV and Parquet files to an S3 bucket

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

### Connecting to the EMR cluster

```
KEY_FILE="emr-cluster-key-pair.pem"
chmod 400 $KEY_FILE
ssh -i $KEY_FILE hadoop@<EMR NODE>
```

## Performing Apache Iceberg queries on S3 tables with Apache Spark

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

```
```

```
```

## Performing time travel queries on S3 tables

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

```
```

```
```

## Working with AWS Lake Formation permissions

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

```
```

```
```

## Running SQL queries in Amazon Athena

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

```
```

```
```
