# A Gentle Introduction to Generative AI and AI Agents on AWS

## Exploring foundation models in Amazon Bedrock

```
Why are there so many different AI models?
Explain in simple terms, with a few clear examples.
```

```
Expand on the previous answer by providing specific examples and use cases for each model mentioned.
```

## Introduction to AI Agents with Amazon Bedrock and Strands Agents

```
!python --version
```

```
%pip --version
```

```
%pip install strands-agents==1.22.0
%pip install strands-agents-builder==0.1.10
%pip install strands-agents-tools==0.2.19
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
from strands import Agent
agent = Agent()

prompt = """What are AI agents? Provide a concise definition (1–2 sentences) that explains what they are and what they do."""

agent(prompt)
```

```
model_id = "us.anthropic.claude-sonnet-4-6"

from strands import Agent

agent = Agent(model=model_id)
prompt = """What are AI agents? Provide a concise definition (1–2 sentences) that explains what they are and what they do."""

agent(prompt)
```

```
from strands_tools import calculator, current_time
```

```
calculator.calculator("1 + 2")
```

```
current_time.current_time()
```

```
agent = Agent(
    tools=[calculator, current_time]
)
```

```
model_id = "us.anthropic.claude-sonnet-4-6"
from strands import Agent

agent = Agent(
    model=model_id,
    tools=[calculator, current_time]
)
```

```
prompt = """I was born on June 1, 1998, and I am 10 years older than my brother. How old is my brother?"""

agent(prompt)
```

```
mid = "us.anthropic.claude-sonnet-4-6"
```

```
from strands.models.bedrock import BedrockModel
from strands import Agent

model = BedrockModel(model_id=mid)
agent = Agent(model)
```

```
prompt = """What are AI agents? Provide a concise definition (1–2 sentences) that explains what they are and what they do."""

agent(prompt)
```

```
agent = Agent(
    model=model, 
    tools=[calculator, current_time]
)
```

```
prompt = """I was born on June 1, 1998, and I am 10 years older than my brother. How old is my brother?"""
```

```
agent(prompt)
```
