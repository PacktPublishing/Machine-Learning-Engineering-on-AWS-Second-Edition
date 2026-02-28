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

## Performing time travel queries on S3 tables

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
