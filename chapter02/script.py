import boto3
from strands import Agent, tool
from strands_tools import calculator, current_time, shell
from strands.models.sagemaker import SageMakerAIModel


REGION = 'us-east-1'
KNOWLEDGEBASE_ID = '<SPECIFY KNOWLEDGEBASE ID>'
ENDPOINT_NAME = 'sagemaker-endpoint-00'


SYSTEM_PROMPT = (
    "You are a RAG agent. When a user asks you a question you will first check it in your knowledge base (if you can't answer it from the current conversation memory)."
    "You MUST use as many tools as possible to answer the user's question."
    "You MUST use the shell tool when working with files and directories."
    "Do not answer without calling a tool first."
)

prompt = """
Get the current month and then tell me more about the Xironal Flux Displacement and Subatomic Event Polarization stored in the knowledge base which happened during the same quarter the last few years. Then, using the generated output, create 1 directory for each month of the said quarter (for example, JAN_2025) and create a text file (for example, info.txt) inside each directory that contains 1-2 sentences relevant to what happened that month. Make sure that the number of directories created is equal to the number of months in the quarter (for example, only 3 months in a quarter).

I am expecting the following file and folder structure to look like this:

JAN_2024/
    info.txt

FEB_2024/
    info.txt
"""

@tool
def search_vector_database(query: str) -> str:
    runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)

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
        
        results = list(filter(
            None, map(lambda r: r.get('content', {}).get('text', ''), 
            response.get('retrievalResults', []))))
        
        return "\n\n".join(results) or "Nothing relevant was found."
    except Exception as e:        
        return f"Error: {str(e)}"


def get_model():
    EPConfig = SageMakerAIModel.SageMakerAIEndpointConfig
    PLConfig = SageMakerAIModel.SageMakerAIPayloadSchema

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


def main():
    agent = Agent(
        model=get_model(),
        tools=[search_vector_database, current_time, shell], 
        system_prompt=SYSTEM_PROMPT
    )

    result = agent(prompt)

    print(result)


if __name__ == "__main__":
    main()
