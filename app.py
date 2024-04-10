import os
import boto3
import uuid
import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
from typing import Optional
import logging
import traceback
import app_bedrock
import app_retrieve_generate
import app_retrieve

AWS_REGION = os.environ["AWS_REGION"]
AUTH_ADMIN_USR = os.environ["AUTH_ADMIN_USR"]
AUTH_ADMIN_PWD = os.environ["AUTH_ADMIN_PWD"]

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
  else:
    return None

async def setup_settings():

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
        "anthropic.claude-instant-v1",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "amazon.titan-text-express-v1",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1"
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
                initial = 2,
                min = 1,
                max = 5,
                step = 1,
            ),
            Select(
                id = "Mode",
                label = "KnowledgeBase Generation Mode",
                values = ["RetrieveAndGenerate", "Retrieve"],
                initial_index = 1,
            ),
            Switch(id="Strict", label="Retrieve - Limit Answers to KnowledgeBase", initial=True),
            Switch(id="Terse", label="Terse - Terse & Consise Answers", initial=True),
            Select(
                id = "Model",
                label = "Foundation Model",
                values = model_ids,
                initial_index = model_ids.index("anthropic.claude-v2"),
            ),
            Slider(
                id = "Temperature",
                label = "Temperature Analytical vs Creative",
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
        ]
    ).send()

    print("Save Settings: ", settings)

    return settings

@cl.on_settings_update
async def setup_agent(settings):

    print("Setup Agent: ", settings)

    knowledge_base_id = settings["KnowledgeBase"]
    
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
        system_message = "You are a helpful assistant. Unless instructed, omit any preamble and provide straight to the point concise answers."
    )

    application_options = dict (
        option_terse = settings["Terse"]
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

    session_id = str(uuid.uuid4())

    cl.user_session.set("session_id", session_id)
    
    settings = await setup_settings()

    await setup_agent(settings)

    #bedrock_list_models(bedrock)


@cl.on_message
async def main(message: cl.Message):

    session_id = cl.user_session.get("session_id") 
    knowledge_base_id = cl.user_session.get("knowledge_base_id") 
    llm_model_arn = cl.user_session.get("llm_model_arn") 
    mode = cl.user_session.get("mode") 
    #["RetrieveAndGenerate", "Retrieve"],
    if mode == "RetrieveAndGenerate":
        await app_retrieve_generate.main_retrieve_and_generate(message)
    else:
        await app_retrieve.main_retrieve(message)


async def main_retrieve_and_generate(message: cl.Message):

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


async def main_retrieve(message: cl.Message):

    session_id = cl.user_session.get("session_id") 
    knowledge_base_id = cl.user_session.get("knowledge_base_id") 
    llm_model_arn = cl.user_session.get("llm_model_arn") 
    bedrock_model_id = cl.user_session.get("bedrock_model_id")
    inference_parameters = cl.user_session.get("inference_parameters")
    strict = cl.user_session.get("strict")
    kb_retrieve_document_count = cl.user_session.get("kb_retrieve_document_count")

    query = message.content

    msg = cl.Message(content="")

    await msg.send()

    try:

        context_info = ""

        async with cl.Step(name="KnowledgeBase", type="llm", root=False) as step:
            step.input = msg.content

            await step.stream_token(f"\nSearch Max {kb_retrieve_document_count} documents.\n")

            try:

                prompt = f"""\n\nHuman: {query[0:900]}
                Assistant:
                """

                response = bedrock_agent_runtime.retrieve(
                    knowledgeBaseId = knowledge_base_id,
                    retrievalQuery={
                        'text': prompt,
                    },
                    retrievalConfiguration={
                        'vectorSearchConfiguration': {
                            'numberOfResults': kb_retrieve_document_count
                        }
                    }
                )

                reference_elements = []
                for i, retrievalResult in enumerate(response['retrievalResults']):
                    uri = retrievalResult['location']['s3Location']['uri']
                    text = retrievalResult['content']['text']
                    excerpt = text[0:75]
                    score = retrievalResult['score']
                    print(f"{i} RetrievalResult: {score} {uri} {excerpt}")
                    #await msg.stream_token(f"\n{i} RetrievalResult: {score} {uri} {excerpt}\n")
                    context_info += f"<p>${text}</p>\n" #context_info += f"${text}\n"
                    #await step.stream_token(f"\n[{i+1}] score={score} uri={uri} len={len(text)} text={excerpt}\n")
                    await step.stream_token(f"\n[{i+1}] score={score} uri={uri} len={len(text)}\n")
                    reference_elements.append(cl.Text(name=f"[{i+1}] {uri}", content=text, display="inline"))
                
                await step.stream_token(f"\n")
                step.elements = reference_elements

            except Exception as e:
                logging.error(traceback.format_exc())
                await step.stream_token(f"{e}")
            finally:
                await step.send()

        async with cl.Step(name="Model", type="llm", root=False) as step_llm:
            step_llm.input = msg.content
            #step_llm.output = f"Test"

            elements = []

            try:

                bedrock_model_strategy = app_bedrock.BedrockModelStrategyFactory.create(bedrock_model_id)

                extra_instructions = ""
                if strict == True:
                    #  Do not answer any question that cannot be answered by the provided context, just say it is not in the provided context.
                    # Keep the answer simple.
                    extra_instructions = "If you don't know the answer or it can't be derived from the provided context, just say that you don't know, and don't try to make up an answer."

                prompt = f"""Use the following pieces of context to answer the user's question. {extra_instructions}
                Here is the context: <context>{context_info}</context>
                \n\nHuman: {query}

                Assistant:
                """
                elements.append(cl.Text(name=f"prompt", content=prompt.replace("\n\n", "").rstrip(), display="inline"))

                max_tokens = inference_parameters.get("max_tokens_to_sample")
                temperature = inference_parameters.get('temperature')
                await step_llm.stream_token(f"model.id={bedrock_model_id} prompt.len={len(prompt)} temperature={temperature} max_tokens={max_tokens}")

                request = bedrock_model_strategy.create_request(inference_parameters, prompt)
 
                print(f"{type(request)} {request}")

                response = bedrock_model_strategy.send_request(request, bedrock_runtime, bedrock_model_id)

                await bedrock_model_strategy.process_response(response, msg)

                step_llm.elements = elements

            except Exception as e:
                logging.error(traceback.format_exc())
                await step_llm.stream_token(f"{e}")
            finally:
                await step_llm.send()

        await msg.stream_token(f". strict={strict}\n")
        await msg.send()

    except Exception as e:
        logging.error(traceback.format_exc())
        await msg.stream_token(f"{e}")
    finally:
        await msg.send()
