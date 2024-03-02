import os
import boto3
import uuid
import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
from typing import Optional
import logging
import traceback

aws_region = os.environ["AWS_REGION"]
AWS_REGION = os.environ["AWS_REGION"]
#aws_profile = os.environ["AWS_PROFILE"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]

knowledge_base_id = os.environ["BEDROCK_KB_ID"]

bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=aws_region)


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
            Switch(id="Generate", label="Retrieve & Generate", initial=True),
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

    knowledge_base_id = settings["KnowledgeBase"]
    
    llm_model_arn = "arn:aws:bedrock:{}::foundation-model/{}".format(AWS_REGION, settings["Model"])

    cl.user_session.set("llm_model_arn", llm_model_arn)
    cl.user_session.set("knowledge_base_id", knowledge_base_id)
    

def bedrock_list_models(bedrock):
    response = bedrock.list_foundation_models(byOutputModality="TEXT")

    for item in response["modelSummaries"]:
        print(item['modelId'])

@cl.on_chat_start
async def main():

    session_id = str(uuid.uuid4())

    cl.user_session.set("session_id", session_id)
    
    settings = await setup_settings()

    await setup_agent(settings)


@cl.on_message
async def main(message: cl.Message):

    session_id = cl.user_session.get("session_id") 
    knowledge_base_id = cl.user_session.get("knowledge_base_id") 
    llm_model_arn = cl.user_session.get("llm_model_arn") 

    query = message.content
    query = query[0:900]

    prompt = f"""\n\nHuman: {query}
    Assistant:
    """
    msg = cl.Message(content="")

    await msg.send()

    try:

        response = bedrock_agent_runtime.retrieve_and_generate(
            #sessionId = session_id,
            input = {
                'text': prompt,
            },
            retrieveAndGenerateConfiguration = {
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': llm_model_arn,
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            #'numberOfResults': 3, # The numberOfResults field is currently unsupported for RetrieveAndGenerate.
                            #'overrideSearchType': 'HYBRID'|'SEMANTIC'
                        }
                    }
                }
            }
        )

        text = response['output']['text']
        await msg.stream_token(text)

        if "citations" in response:
            #print(response["citations"])
            for citation in response["citations"]:
                if "retrievedReferences" in citation:
                    references = citation["retrievedReferences"]
                    #print(references)
                    reference_idx = 1
                    for reference in references:
                        reference_idx = reference_idx + 1
                        print(reference)
                        reference_text = ""
                        reference_location_type = ""
                        reference_location_uri = ""
                        if "content" in reference:
                            content = reference["content"]
                            reference_text = content["text"]
                        if "location" in reference:
                            location = reference["location"]
                            location_type = location["type"]
                            reference_location_type = location_type
                            if "S3" == location_type:
                                location_uri = location["s3Location"]["uri"]
                                reference_location_uri = location_uri
                        await msg.stream_token(f"\n{reference_location_type} {reference_location_uri}")
                        #elements.append(cl.Text(name=f"src_{reference_idx}", content=f"{location_uri}\n{reference_text}"))

    except Exception as e:
        logging.error(traceback.format_exc())
        await msg.stream_token(f"{e}")
    finally:
        await msg.send()