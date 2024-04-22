import os
import boto3
import chainlit as cl
import logging
import traceback
import app_bedrock

AWS_REGION = os.environ["AWS_REGION"]

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

async def create_prompt(application_options: dict, query: str) -> str:

        option_terse = application_options.get("option_terse")
        option_strict = application_options.get("option_strict")

        strict_instructions = ""
        if option_strict == True:
            strict_instructions = "Only answer if you know the answer with certainty. Otherwise, just say that you don't know and don't make up an answer."

        terse_instructions = ""
        if option_terse == True:
            terse_instructions = "Unless otherwise instructed, omit any preamble and provide terse and concise one liner answer."

        prompt = f"""Please answer the question while following instructions provided. {terse_instructions} {strict_instructions}
        \n\nHuman: {query}

        Assistant:
        """

        return prompt

async def main_retrieve(message: cl.Message):

    application_options = cl.user_session.get("application_options")
    session_id = cl.user_session.get("session_id") 
    knowledge_base_id = cl.user_session.get("knowledge_base_id") 
    llm_model_arn = cl.user_session.get("llm_model_arn") 
    bedrock_model_id = cl.user_session.get("bedrock_model_id")
    inference_parameters = cl.user_session.get("inference_parameters")
    strict = cl.user_session.get("strict")
    option_terse = application_options.get("option_terse")
    kb_retrieve_document_count = cl.user_session.get("kb_retrieve_document_count")

    query = message.content

    msg = cl.Message(content="")

    await msg.send()

    try:

        async with cl.Step(name="Model", type="llm", root=False) as step_llm:
            step_llm.input = msg.content

            elements = []

            try:

                bedrock_model_strategy = app_bedrock.BedrockModelStrategyFactory.create(bedrock_model_id)

                prompt = await create_prompt(application_options, query)

                system_message = inference_parameters.get("system_message")
                elements.append(cl.Text(name=f"system", content=system_message.replace("\n\n", "").rstrip(), display="inline")) 

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

        await msg.stream_token(f". strict={strict} terse={option_terse}\n")
        await msg.send()

    except Exception as e:
        logging.error(traceback.format_exc())
        await msg.stream_token(f"{e}")
    finally:
        await msg.send()

