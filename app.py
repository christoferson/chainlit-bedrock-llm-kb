import os
import boto3
import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
from typing import Optional
import app_retrieve_generate
import app_retrieve
import app_generate
import app_bedrock_lib
from typing import List

AWS_REGION = os.environ["AWS_REGION"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]
AUTH_USER_USR = os.environ["AUTH_USER_USR"]
AUTH_USER_PWD = os.environ["AUTH_USER_PWD"]

bedrock = boto3.client("bedrock", region_name=AWS_REGION)
bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent = boto3.client('bedrock-agent', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database
  if (username, password) == (AUTH_ADMIN_USR, AUTH_ADMIN_PWD):
    return cl.User(identifier=AUTH_ADMIN_USR, metadata={"role": "admin", "provider": "credentials"})
  elif (username, password) == (AUTH_USER_USR, AUTH_USER_PWD):
    return cl.User(identifier=AUTH_USER_USR, metadata={"role": "user", "provider": "credentials"})
  else:
    return None

async def setup_settings():

    kb_id_list = await app_bedrock_lib.list_knowledge_bases()

    model_ids = [
        "anthropic.claude-v2", #"anthropic.claude-v2:0:18k",
        "anthropic.claude-instant-v1",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "amazon.titan-text-express-v1",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "cohere.command-light-text-v14",
        "cohere.command-text-v14",
        #"meta.llama2-13b-chat-v1",
        #"meta.llama2-70b-chat-v1",
        "ai21.j2-mid"
    ]

    settings = await cl.ChatSettings(
        [
            Select(
                id="KnowledgeBase",
                label="KnowledgeBase ID",
                values=kb_id_list,
                initial_index=0
            ),
            Slider(
                id = "RetrieveDocumentCount",
                label = "KnowledgeBase DocumentCount",
                initial = 8,
                min = 1,
                max = 24,
                step = 1,
            ),
            Select(
                id = "Mode",
                label = "KnowledgeBase Generation Mode",
                values = ["RetrieveAndGenerate", "Retrieve", "Generate"],
                initial_index = 1,
            ),
            Select(
                id = "Model",
                label = "Foundation Model",
                values = model_ids,
                initial_index = model_ids.index("anthropic.claude-3-sonnet-20240229-v1:0"),
            ),
            Slider(
                id = "Temperature",
                label = "Temperature - Analytical vs Creative",
                initial = 0.0,
                min = 0,
                max = 1,
                step = 0.1,
            ),
            Slider(
                id = "TopP",
                label = "Top P - High Diversity vs Low Diversity",
                initial = 1,
                min = 0,
                max = 1,
                step = 0.1,
            ),
            Slider(
                id = "TopK",
                label = "Top K - High Probability vs Low Probability",
                initial = 250,
                min = 0,
                max = 1500,
                step = 5,
            ),
            Slider(
                id="MaxTokenCount",
                label="Max Token Size",
                initial = 3072,
                min = 256,
                max = 4096,
                step = 256,
            ),
            Switch(id="Strict", label="Retrieve - Limit Answers to KnowledgeBase", initial=False),
            Switch(id="Terse", label="Terse - Terse & Consise Answers", initial=False),
            Switch(id="AlignUnitsOfMeasure", label="Align Units of Measure", initial=True),
            Switch(id="AlignGeographicDivisions", label="Align Geographic Divisions", initial=True),
            Switch(id="StateConclusionFirst", label="Always State Conclusions First", initial=False),
            Switch(id="SourceTableMarkdown", label="Source Tables Markdown Display", initial=True),
        ]
    ).send()

    print("Save Settings: ", settings)

    return settings

@cl.on_settings_update
async def setup_agent(settings):

    print("Setup Agent: ", settings)

    knowledge_base_id = settings["KnowledgeBase"]
    knowledge_base_id = knowledge_base_id.split(" ", 1)[0]
    
    llm_model_arn = "arn:aws:bedrock:{}::foundation-model/{}".format(AWS_REGION, settings["Model"])
    mode = settings["Mode"]
    strict = settings["Strict"]
    kb_retrieve_document_count = int(settings["RetrieveDocumentCount"])

    bedrock_model_id = settings["Model"]

    inference_parameters = dict (
        temperature = settings["Temperature"],
        top_p = float(settings["TopP"]),
        top_k = int(settings["TopK"]),
        max_tokens_to_sample = int(settings["MaxTokenCount"]),
        stop_sequences =  [],
        #system_message = "You are a helpful assistant. Unless instructed, omit any preamble and provide straight to the point concise answers."
        system_message = "You are a helpful assistant and tries to answer questions as best as you can."
    )

    application_options = dict (
        option_terse = settings["Terse"],
        option_strict = settings["Strict"],
        option_align_units_of_measure = settings["AlignUnitsOfMeasure"],
        option_align_geographic_divisions = settings["AlignGeographicDivisions"],
        option_state_conclusions_first = settings["StateConclusionFirst"],
        option_source_table_markdown_display = settings["SourceTableMarkdown"]
    )

    cl.user_session.set("inference_parameters", inference_parameters)
    cl.user_session.set("bedrock_model_id", bedrock_model_id)
    cl.user_session.set("llm_model_arn", llm_model_arn)
    cl.user_session.set("knowledge_base_id", knowledge_base_id)
    cl.user_session.set("kb_retrieve_document_count", kb_retrieve_document_count)
    cl.user_session.set("mode", mode)
    cl.user_session.set("strict", strict)
    cl.user_session.set("application_options", application_options)
    
    

def bedrock_list_models(bedrock):
    response = bedrock.list_foundation_models(byOutputModality="TEXT")

    for item in response["modelSummaries"]:
        print(item['modelId'])

@cl.on_chat_start
async def main():

    #session_id = str(uuid.uuid4())
    #cl.user_session.set("session_id", session_id)

    settings = await setup_settings()

    await setup_agent(settings)

    #bedrock_list_models(bedrock)


@cl.on_message
async def main(message: cl.Message):

    mode = cl.user_session.get("mode") 

    if mode == "RetrieveAndGenerate":
        await app_retrieve_generate.main_retrieve_and_generate(message)
    elif mode == "Generate":
        await app_generate.main_retrieve(message)
    else:
        await app_retrieve.main_retrieve(message)
