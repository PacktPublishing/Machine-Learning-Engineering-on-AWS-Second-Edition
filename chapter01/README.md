# Chapter 1: A Gentle Introduction to Generative AI and AI Agents on AWS

This README.md file contains the commands and code snippets referenced in a chapter of *Machine Learning Engineering on AWS — Second Edition* by Joshua Arvin Lat, published by Packt. It is intended to support the examples in the book by making it simpler for you to copy, run, and modify the provided materials. 

![Machine Learning Engineering on AWS — Second Edition](../book-cover.png)

To help you get started more easily, the repository includes a [DETAILS.md](https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md) file containing additional guidance, references, and important notes for the examples discussed throughout the book.

## Technical Requirements

We must have the following ready before we jump into the hands-on examples of this chapter:

- **A code editor installed on your local machine (such as Visual Studio Code or Sublime Text)**: You'll need this when working with the code and configuration files used throughout the hands-on exercises and examples in this book.
- **A new AWS account**: Creating and using a new AWS account is highly recommended for the hands-on exercises and solutions in this book. This will ensure that your work-related production (or staging) environment resources remain separate and secure. Feel free to create this AWS account by going to https://aws.amazon.com/free/. On this page, locate and click on the Create an AWS Account button to proceed with the creation of the account. Make sure to read the available AWS documentation along with the relevant FAQ pages to have a better awareness of what is free (and what is not free) when using various services in AWS.


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


## Where to Get Your Copy

You can grab your copy of *Machine Learning Engineering on AWS — Second Edition* through popular online retailers such as [Amazon](https://amazon.com/author/arvs) or directly from the publisher, [Packt](https://www.packtpub.com/en-us/product/machine-learning-engineering-on-aws-9781835881088). Feel free to choose the format that works best for you. 🙏
