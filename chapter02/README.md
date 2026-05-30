# Chapter 2: Building AI Agents with SageMaker AI and Bedrock AgentCore

In this chapter, you'll learn how to build AI agents that interact with a SageMaker AI real-time inference endpoint. You will use Amazon Bedrock Knowledge Bases and Amazon S3 Vectors to build retrieval-augmented generation powered agents, while also exploring how Strands Agents and Bedrock AgentCore can integrate model inference, external tools, and knowledge retrieval into production-ready agent-based systems.

To help you gain hands-on experience running AI agents with SageMaker AI, Strands Agents, and Bedrock AgentCore, we will cover the following topics in this chapter:

- Deploying a pretrained LLM with SageMaker AI
- Building AI agents with Amazon SageMaker AI and Strands Agents
- Building AI agents with Amazon Bedrock AgentCore
- Deploying production-ready agents with Bedrock AgentCore Runtime
- Setting up an Amazon Bedrock Knowledge Base
- Building a RAG-powered AI agent with Strands Agents
- Building a RAG-powered AI agent that interacts with a SageMaker AI inference endpoint

This README.md file contains the commands and code snippets referenced in a chapter of *Machine Learning Engineering on AWS — Second Edition* by Joshua Arvin Lat, published by Packt. It is intended to support the examples in the book by making it simpler for you to copy, run, and modify the provided materials. 

![Machine Learning Engineering on AWS 2nd ed](../books.png)

To help you get started more easily, the repository includes a [DETAILS.md](https://github.com/PacktPublishing/Machine-Learning-Engineering-on-AWS-Second-Edition/blob/main/DETAILS.md) file containing additional guidance, references, and important notes for the examples discussed throughout the book.

## Technical Requirements

Before proceeding with the hands-on examples in this chapter, ensure that the following prerequisites and setup requirements are in place:

- **Sufficient account-level quota for selected ML instance types**: When deploying models, ensure that the applied account-level quota value for the selected ML instance types (for example, ml.g5.24xlarge for endpoint usage) is set to at least 1. If the quota value is 0, SageMaker AI will not be able to provision the required compute resources, and the job or endpoint creation will fail. You can review and request quota increases through the Service Quotas console (within the AWS Management Console).
- **An existing SageMaker Studio space**: You can use the SageMaker Studio space (mle-on-aws-space) that you set up in Chapter 1 to follow along with the examples in this book.
- **A code editor installed on your local machine (such as Visual Studio Code or Sublime Text)**: You'll need this when working with the code and configuration files used throughout the hands-on exercises and examples in this book.
- **A new IAM user with AdministratorAccess permissions**: You can create this IAM user through the IAM Console, by typing IAM in the search bar, and then selecting IAM from the list of results. Once in the IAM console, select Users from the left-hand menu and then click the Create user button. Enter the desired user name (superadmin), ensure that the checkbox for Provide user access to the AWS Management Console is checked, choose I want to create an IAM user, and click Next. On the Set permissions page, choose Attach policies directly, then search for and select the AdministratorAccess policy. Continue through the remaining steps, and finally clicking the Create user button. After completion, you'll be able to use the new IAM user to log in to the AWS Management Console with full administrative permissions.

| Note |
|:-----|
| You will need to sign in using this IAM user account (superadmin) when setting up the Amazon Bedrock knowledge base. Keep in mind that this IAM user has overly permissive permissions (AdministratorAccess), which grants full access to all AWS services and resources. It's recommended to create more restricted IAM users in production environments to follow the principle of least privilege. For now, this should do the trick, so that you can work through the hands-on examples without running into permission issues. |

## Deploying a Pretrained LLM with SageMaker AI

```
%pip install 'strands-agents[sagemaker]'
```

```
%pip uninstall -y sagemaker sagemaker-serve
%pip install "sagemaker==3.2.0"
%pip install "sagemaker-serve==1.2.0"
```

```
%pip show sagemaker
```

```
%pip show sagemaker-serve
```

```
import sagemaker
```

```
sagemaker.__file__
```

```
from import_helper import ImportHelper
```

```
helper = ImportHelper(sagemaker)
helper.explore()
```

```
helper.guess_import("ModelBuilder")
```

```
import sagemaker.serve.model_builder as MB
import sagemaker.serve.builder.schema_builder as SB
ModelBuilder = MB.ModelBuilder
SchemaBuilder = SB.SchemaBuilder
```

```
from sagemaker.core.helper.session_helper import (
    Session
)

session = Session()
bucket_name = session.default_bucket() 
role_arn = session.get_caller_identity_arn()
```

```
model_id = "huggingface-reasoning-qwen3-32b"
instance_type = "ml.g5.24xlarge"
```

```
example_input = {
    "inputs": "AI agents are systems that can",
    "parameters": {
        "max_new_tokens": 256,
        "top_p": 0.95,
        "temperature": 0.7
    }
}
```

```
example_output = [
    {"generated_text": """AI agents are systems that can perceive their environment, reason about goals, and take actions autonomously."""}
]
```

```
schema_builder = SB.SchemaBuilder(
    example_input, 
    example_output
)
```

```
env_vars = {
    "OPTION_TOOL_CALL_PARSER": "hermes",
    "OPTION_ENABLE_AUTO_TOOL_CHOICE": "true",
}
```

```
import logging
log_level = logging.ERROR

model_builder = MB.ModelBuilder( 
    model=model_id, 
    schema_builder=schema_builder,
    sagemaker_session=session, 
    role_arn=role_arn, 
    log_level=log_level,
    instance_type=instance_type,
    env_vars=env_vars
)
```

```
model = model_builder.build()
```

```
from sagemaker.core.shapes.shapes import (
    ModelAccessConfig
)

model_access_configs = {
    model_id: ModelAccessConfig(
        accept_eula=True
    )
}
```

```
%%time

predictor = model_builder.deploy(
    endpoint_name="sagemaker-endpoint-00",
    model_access_configs=model_access_configs, 
    accept_eula=True
)
```

```
print(example_input)
```

```
import json

resp = predictor.invoke(
    body=json.dumps(example_input),
    content_type="application/json"
)

print(resp.body.read())
```

## Building AI Agents with Amazon SageMaker AI and Strands Agents

```
%pip install strands-agents==1.22.0
%pip install strands-agents-builder==0.1.10
%pip install 'strands-agents[sagemaker]'
%pip install strands-agents-tools==0.2.19
```

```
%pip show strands-agents
```

```
%pip show strands-agents-builder
```

```
%pip show strands-agents-tools
```

```
import strands
from import_helper import ImportHelper

helper = ImportHelper(strands)
helper.guess_import("SageMakerAIModel")
```

```
endpoint_name="sagemaker-endpoint-00"
```

```
from strands.models.sagemaker import SageMakerAIModel

EPConfig = SageMakerAIModel.SageMakerAIEndpointConfig
PLConfig = SageMakerAIModel.SageMakerAIPayloadSchema

endpoint_config = EPConfig(
    endpoint_name=endpoint_name, 
    region_name="us-east-1"
)

payload_config = PLConfig(
    max_tokens=4096, 
    stream=True
)
```

```
model = SageMakerAIModel(
    endpoint_config=endpoint_config,
    payload_config=payload_config
)
```

```
from strands import Agent
from strands_tools import calculator, current_time

system_prompt = (
    "You MUST use at least one tool to answer the user's question."
    "Do not answer without calling a tool first."
)

agent = Agent(
    model=model, 
    tools=[calculator, current_time],
    system_prompt=system_prompt
)
```

```
%%time

prompt = """I was born on June 1, 1998, and I am 10 years older than my brother. How old is my brother?"""

response = agent(prompt)
```

## Building AI Agents with Amazon Bedrock AgentCore

### Getting Started with Amazon Bedrock AgentCore

```
which uv
```

```
pip install bedrock-agentcore-starter-toolkit==0.2.7
```

```
agentcore create --project-name helloSageMaker
```

```
cd helloSageMaker/
```

```
source .venv/bin/activate
```

```
sudo apt install -y tree
```

```
tree
```

```
agentcore dev
```

```
agentcore invoke --dev "Hello Agent!"
```

### Customizing your agent to interact with a SageMaker AI endpoint

```
...

dependencies = [
    ...
    "strands-agents-tools >= 0.2.16",
    "strands-agents[sagemaker]",
    "mypy-boto3-sagemaker-runtime",
]
```

```
uv sync
```

```
import os
from strands import Agent
from strands_tools import calculator
from strands.models.sagemaker import SageMakerAIModel

from bedrock_agentcore.runtime import (
    BedrockAgentCoreApp
)
from mcp_client.client import (
    get_streamable_http_mcp_client
)
```

```
app = BedrockAgentCoreApp()
log = app.logger

REGION = os.getenv("AWS_REGION")
mcp_client = get_streamable_http_mcp_client()

SAIModel = SageMakerAIModel
EndpointConfig = SAIModel.SageMakerAIEndpointConfig
PayloadConfig = SAIModel.SageMakerAIPayloadSchema
```

```
SYSTEM_PROMPT=(
    "You MUST use at least one tool to answer the user's question."
    "Do not answer without calling a tool first."
)
```

```
def load_sagemaker_model(endpoint_name="sagemaker-endpoint-00"):
    endpoint_config = EndpointConfig(
        endpoint_name=endpoint_name, 
        region_name="us-east-1"
    )
    
    payload_config = PayloadConfig(
        max_tokens=4096, stream=True
    )
    
    model = SageMakerAIModel(
        endpoint_config=endpoint_config,
        payload_config=payload_config
    )

    return model
```

```
@app.entrypoint
async def invoke(payload, context):
    with mcp_client as client:
        agent = Agent(
            model=load_sagemaker_model(), 
            tools=[calculator, current_time],
            system_prompt=SYSTEM_PROMPT
        )

        stream = agent.stream_async(
            payload.get("prompt")
        )

        async for event in stream:
            if "data" in event and isinstance(
                event["data"], str
            ):
                yield event["data"]
```

```
if __name__ == "__main__":
    app.run()
```

```
agentcore dev
```

```
agentcore invoke --dev "I was born on June 1, 1998, and I am 10 years older than my brother. How old is my brother?"
```

## Deploying Production-Ready Agents with Bedrock AgentCore Runtime

### Deploying an Agent with Bedrock AgentCore Runtime

```
sudo apt install -y zip
```

```
agentcore deploy
```

```
agentcore invoke "I was born on June 1, 1998, and I am 10 years older than my brother. How old is my brother?"
```

```
agentcore invoke "I was born on June 1, 1998, and I am 10 years older than my brother. How old is my brother?"
```

### Cleaning up

```
agentcore destroy
```

## Setting up an Amazon Bedrock Knowledge Base

```
Xironal Flux Displacement
```

```
When was the Xironal Flux Displacement phenomenon discovered?
```

## Building a RAG-powered AI agent with Strands Agents

```
%pip install strands-agents==1.22.0
%pip install strands-agents-builder==0.1.10
%pip install strands-agents-tools==0.2.19
%pip install 'strands-agents[sagemaker]'
%pip install strands-agents-tools
```

```
REGION='<SPECIFY REGION>'
KNOWLEDGEBASE_ID='<SPECIFY KNOWLEDGE BASE ID>'
```

```
import boto3
runtime = boto3.client(
    "bedrock-agent-runtime", 
    region_name=REGION
)
```

```
query = "Xironal Flux Displacement"

retrieval_config = {                
    "vectorSearchConfiguration": {                    
        "numberOfResults": 3
    }
}

response = runtime.retrieve(            
    knowledgeBaseId=KNOWLEDGEBASE_ID,            
    retrievalQuery={"text": query},            
    retrievalConfiguration=retrieval_config
)
```

```
response['retrievalResults']
```

```
m = map(
    lambda r: r.get(
        'content', {}
    ).get('text', ''), 
    response.get('retrievalResults', [])
)

results = list(filter(None, m))
```

```
results
```

```
from strands import tool

@tool
def search_vector_database(query: str) -> str:
    """    
    Use the knowledge base to find a specific topic by searching the knowledge base for relevant context to answer questions.
    Args:
        query: A question about concepts or terminology  
    Returns:
        A formatted string response generated from the knowledge base
    """
    
    runtime = boto3.client("bedrock-agent-runtime", 
                           region_name=REGION)

    try:        
        response = runtime.retrieve(            
            knowledgeBaseId=KNOWLEDGEBASE_ID,            
            retrievalQuery={"text": query},            
            retrievalConfiguration={                
                "vectorSearchConfiguration": {                    
                    "numberOfResults": 3
                }
            }
        )

        m = map(
            lambda r: r.get(
                'content', {}
            ).get('text', ''), 
            response.get('retrievalResults', [])
        )
        
        results = list(filter(None, m))
        
        return "\n\n".join(results) or "No results."
    except Exception as e:        
        return f"Error: {str(e)}"
```

```
search_vector_database("Xironal Flux Displacement")
```

```
from strands.models.bedrock import BedrockModel

mid = "us.anthropic.claude-sonnet-4-6"
model = BedrockModel(model_id=mid)
```

```
from strands import Agent
from strands_tools import calculator, current_time

SYSTEM_PROMPT= (
    "You are a RAG agent. When a user asks you a question you will first check it in your knowledge base (if you can't answer it from the current conversation memory)."
    "You MUST use at least one tool to answer the user's question."
    "Do not answer without calling a tool first."
)

agent = Agent(
    model=model,
    tools=[search_vector_database, current_time], 
    system_prompt=SYSTEM_PROMPT
)
```

```
prompt = """Get the current month and then tell me more about the Xironal Flux Displacement and Subatomic Event Polarization stored in the knowledge base which happened during the same quarter the last few years"""
result = agent(prompt)
```

## Building a RAG-powered AI agent that interacts with a SageMaker AI inference endpoint

### Building the RAG-powered agent

```
import boto3
from strands import Agent, tool
from strands_tools import (
    current_time, 
    shell
)
from strands.models.sagemaker import SageMakerAIModel
```

```
REGION = 'us-east-1'
KNOWLEDGEBASE_ID = '<SPECIFY KNOWLEDGEBASE ID>'
ENDPOINT_NAME = 'sagemaker-endpoint-00'
```

```
SYSTEM_PROMPT = (
    "You are a RAG agent. When a user asks you a question you will first check it in your knowledge base (if you can't answer it from the current conversation memory)."
    "You MUST use as many tools as possible to answer the user's question."
    "You MUST use the shell tool when working with files and directories."
    "Do not answer without calling a tool first."
)
```

```
prompt = """
Get the current month and then tell me more about the Xironal Flux Displacement and Subatomic Event Polarization stored in the knowledge base which happened during the same quarter the last few years. Then, using the generated output, create 1 directory for each month of the said quarter (for example, JAN_2025) and create a text file (for example, info.txt) inside each directory that contains 1-2 sentences relevant to what happened that month. Make sure that the number of directories created is equal to the number of months in the quarter (for example, only 3 months in a quarter).

I am expecting the following file and folder structure to look like this:

JAN_2024/
    info.txt

FEB_2024/
    info.txt
"""
```

```
@tool
def search_vector_database(query: str) -> str:
    runtime = boto3.client(
        "bedrock-agent-runtime", 
        region_name=REGION
    )

    try:        
        response = runtime.retrieve(            
            knowledgeBaseId=KNOWLEDGEBASE_ID,            
            retrievalQuery={"text": query},            
            retrievalConfiguration={                
                "vectorSearchConfiguration": {                    
                    "numberOfResults": 3
                }
            }
        )
        
        m = map(
            lambda r: r.get(
                'content', {}
            ).get('text', ''), 
            response.get('retrievalResults', [])
        )
        
        results = list(filter(None, m))
        
        return "\n\n".join(results) or "No results."
    except Exception as e:        
        return f"Error: {str(e)}"
```

```
def get_model():
    SM = SageMakerAIModel
    EPConfig = SM.SageMakerAIEndpointConfig
    PLConfig = SM.SageMakerAIPayloadSchema

    endpoint_config = EPConfig(
        endpoint_name=ENDPOINT_NAME, 
        region_name=REGION
    )

    payload_config = PLConfig(
        max_tokens=4096, 
        stream=True
    )

    model = SageMakerAIModel(
        endpoint_config=endpoint_config,
        payload_config=payload_config
    )

    return model
```

```
def main():
    agent = Agent(
        model=get_model(),
        tools=[
            search_vector_database, 
            current_time, 
            shell
        ], 
        system_prompt=SYSTEM_PROMPT
    )

    result = agent(prompt)
    print(result)
```

```
if __name__ == "__main__":
    main()
```

```
python main.py
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
