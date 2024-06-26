import os
import boto3
import chainlit as cl
import logging
import traceback
import app_bedrock

from botocore.exceptions import ClientError

AWS_REGION = os.environ["AWS_REGION"]

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

async def main_retrieve(message: cl.Message):

    application_options = cl.user_session.get("application_options")
    #session_id = cl.user_session.get("session_id") 
    knowledge_base_id = cl.user_session.get("knowledge_base_id") 
    #llm_model_arn = cl.user_session.get("llm_model_arn") 
    bedrock_model_id = cl.user_session.get("bedrock_model_id")
    inference_parameters = cl.user_session.get("inference_parameters")
    option_strict = application_options.get("option_strict")
    option_terse = application_options.get("option_terse")
    option_align_units_of_measure = application_options.get("option_align_units_of_measure")
    option_state_conclusions_first = application_options.get("option_state_conclusions_first")
    option_source_table_markdown_display = application_options.get("option_source_table_markdown_display")
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
                    context_info += f"{text}\n" #context_info += f"<p>${text}</p>\n" #context_info += f"${text}\n"
                    #await step.stream_token(f"\n[{i+1}] score={score} uri={uri} len={len(text)} text={excerpt}\n")
                    await step.stream_token(f"\n[{i+1}] score={score} uri={uri} len={len(text)}\n")
                    reference_elements.append(cl.Text(name=f"[{i+1}] {uri}", content=text, display="inline"))
                
                await step.stream_token(f"\n")
                step.elements = reference_elements

            except Exception as e:
                logging.error(traceback.format_exc())
                await msg.stream_token(f"{e}")
            finally:
                await step.send()

        async with cl.Step(name="Model", type="llm", root=False) as step_llm:
            step_llm.input = msg.content

            elements = []

            try:

                bedrock_model_strategy = app_bedrock.BedrockModelStrategyFactory.create(bedrock_model_id)

                # Create Prompt create_prompt(self, application_options: dict, context_info: str, query: str) -> str:

                prompt = bedrock_model_strategy.create_prompt(application_options, context_info, query)

                # End - Create Prompt 

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

            except ClientError as err:
                message = err.response["Error"]["Message"]
                logging.error("A client error occurred: %s", message)
                await msg.stream_token(f"{message}")
            except Exception as e:
                logging.error(traceback.format_exc())
                await msg.stream_token(f"{e}")
            finally:
                await step_llm.send()


        await msg.stream_token(f". strict={option_strict} terse={option_terse} align_units={option_align_units_of_measure} conclusions_first={option_state_conclusions_first} markdown={option_source_table_markdown_display}\n")
        await msg.send()

    except Exception as e:
        logging.error(traceback.format_exc())
        await msg.stream_token(f"{e}")
    finally:
        await msg.send()
