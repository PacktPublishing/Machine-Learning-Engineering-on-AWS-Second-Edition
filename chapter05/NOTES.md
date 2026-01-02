<img width="468" height="22" alt="image" src="https://github.com/user-attachments/assets/47a4c48b-492d-46a1-8ee2-bef08cd3522f" />## Technical requirements

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
