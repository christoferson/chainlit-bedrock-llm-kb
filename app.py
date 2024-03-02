import os
import boto3
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
import chainlit as cl
from chainlit.input_widget import Select, Slider
from typing import Optional

aws_region = os.environ["AWS_REGION"]
AWS_REGION = os.environ["AWS_REGION"]
#aws_profile = os.environ["AWS_PROFILE"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]

knowledge_base_id = os.environ["BEDROCK_KB_ID"]

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database
  if (username, password) == (AUTH_ADMIN_USR, AUTH_ADMIN_PWD):
    return cl.User(identifier=AUTH_ADMIN_USR, metadata={"role": "admin", "provider": "credentials"})
  else:
    return None

async def setup_settings():

    bedrock = boto3.client("bedrock", region_name=AWS_REGION)
    bedrock_agent = boto3.client('bedrock-agent', region_name=AWS_REGION)
    
    response = bedrock_agent.list_knowledge_bases(maxResults = 5)

    kb_id_list = []
    for i, knowledgeBaseSummary in enumerate(response['knowledgeBaseSummaries']):
        knowledgeBaseId = knowledgeBaseSummary['knowledgeBaseId']
        name = knowledgeBaseSummary['name']
        description = knowledgeBaseSummary['description']
        status = knowledgeBaseSummary['status']
        updatedAt = knowledgeBaseSummary['updatedAt']
        #print(f"{i} RetrievalResult: {knowledgeBaseId} {name} {description} {status} {updatedAt}")
        kb_id_list.append(knowledgeBaseId)

    model_ids = [
        "anthropic.claude-v2", #"anthropic.claude-v2:0:18k",
        "anthropic.claude-v2:1", #"anthropic.claude-v2:1:18k", "anthropic.claude-v2:1:200k", 
        "anthropic.claude-instant-v1"
    ]

    settings = await cl.ChatSettings(
        [
            Select(
                id="KnowledgeBase",
                label="Amazon Bedrock - KnowledgeBase",
                values=kb_id_list,
                initial_index=0#model_ids.index("anthropic.claude-v2"),
            ),
            Select(
                id = "Model",
                label = "Foundation Model",
                values = model_ids,
                initial_index = model_ids.index("anthropic.claude-v2"),
            ),
            Slider(
                id = "Temperature",
                label = "Temperature",
                initial = 0.0,
                min = 0,
                max = 1,
                step = 0.1,
            ),
            Slider(
                id = "TopP",
                label = "Top P",
                initial = 1,
                min = 0,
                max = 1,
                step = 0.1,
            ),
            Slider(
                id = "TopK",
                label = "Top K",
                initial = 250,
                min = 0,
                max = 500,
                step = 5,
            ),
            Slider(
                id="MaxTokenCount",
                label="Max Token Size",
                initial = 2048,
                min = 256,
                max = 4096,
                step = 256,
            ),
            Slider(
                id = "DocumentCount",
                label = "Document Count",
                initial = 1,
                min = 1,
                max = 5,
                step = 1,
            )
        ]
    ).send()

    print("Save Settings: ", settings)

    return settings

@cl.on_settings_update
async def setup_agent(settings):

    print("Setup Agent: ", settings)

    bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=aws_region)

    llm = BedrockChat(
        client = bedrock_runtime,
        model_id = settings["Model"], 
        model_kwargs = {
            "temperature": settings["Temperature"],
            "top_p": settings["TopP"],
            "top_k": int(settings["TopK"]),
            "max_tokens_to_sample": int(settings["MaxTokenCount"]),
        },
        streaming = True
    )

    message_history = ChatMessageHistory()
    
    retriever = MyAmazonKnowledgeBasesRetriever(
        client = bedrock_agent_runtime,
        knowledge_base_id = knowledge_base_id,
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": settings["DocumentCount"]
            }
        }
    )

    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        output_key = "answer",
        chat_memory = message_history,
        return_messages = True,
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type = "stuff",
        retriever = retriever,
        memory = memory,
        return_source_documents = True,
        verbose = True,
       # max_tokens_limit=1000,
    )

    # Store the chain in the user session
    cl.user_session.set("chain", chain)

def bedrock_list_models(bedrock):
    response = bedrock.list_foundation_models(byOutputModality="TEXT")

    for item in response["modelSummaries"]:
        print(item['modelId'])

@cl.on_chat_start
async def main():

    ##
    #print(f"Profile: {aws_profile} Region: {aws_region}")
    bedrock = boto3.client("bedrock", region_name=aws_region)
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=aws_region)

    #bedrock_list_models(bedrock)

    ##
        
    settings = await setup_settings()

    await setup_agent(settings)


@cl.on_message
async def main(message: cl.Message):

    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    res = await chain.ainvoke(
        message.content, 
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            #print(source_doc.metadata['location']) # {'type': 'S3', 's3Location': {'uri': 's3://xxx'}}
            source_content = f"{source_doc.metadata['score']} | {source_doc.page_content}"
            text_elements.append(
                cl.Text(content=source_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()



###
    
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
    
class MyAmazonKnowledgeBasesRetriever(AmazonKnowledgeBasesRetriever):

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        print("---------------------------------")
        nquery = query[0:998]
        return AmazonKnowledgeBasesRetriever._get_relevant_documents(self, query=nquery, run_manager=run_manager)