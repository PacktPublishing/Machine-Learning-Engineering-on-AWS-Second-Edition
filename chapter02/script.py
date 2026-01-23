import boto3
from strands import Agent, tool
from strands_tools import calculator, current_time, shell
from strands.models.sagemaker import SageMakerAIModel

@tool
def search_vector_database(query: str, region: str, knowledgebase_id: str) -> str:
    runtime = boto3.client("bedrock-agent-runtime", region_name=region)

    try:        
        response = runtime.retrieve(            
            knowledgeBaseId=knowledgebase_id,            
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

def main():
    region = input("Enter the AWS region (e.g., us-east-1): ")
    knowledgebase_id = input("Enter the Knowledge Base ID: ")
    endpoint_name = input("Enter the SageMaker endpoint name: ")

    EPConfig = SageMakerAIModel.SageMakerAIEndpointConfig
    PLConfig = SageMakerAIModel.SageMakerAIPayloadSchema

    endpoint_config = EPConfig(
        endpoint_name=endpoint_name, 
        region_name=region
    )

    payload_config = PLConfig(
        max_tokens=4096, 
        stream=True
    )

    model = SageMakerAIModel(
        endpoint_config=endpoint_config,
        payload_config=payload_config
    )

    SYSTEM_PROMPT = (
        "You are a RAG agent. When a user asks you a question you will first check it in your knowledge base (if you can't answer it from the current conversation memory)."
        "You MUST use as many tools as possible to answer the user's question."
        "You MUST use the shell tool when working with files and directories."
        "Do not answer without calling a tool first."
    )

    agent = Agent(
        model=model,
        tools=[search_vector_database, current_time, shell], 
        system_prompt=SYSTEM_PROMPT
    )

    prompt = """Get the current month and then tell me more about the Xironal Flux Displacement and Subatomic Event Polarization stored in the knowledge base which happened during the same quarter the last few years. Then, using the generated output, create 1 directory per month and create a text file inside each directory that contains information relevant to that month."""

    result = agent(prompt)

    print(result)

if __name__ == "__main__":
    main()
