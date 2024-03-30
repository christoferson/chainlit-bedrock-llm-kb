import os
import boto3
import chainlit as cl
import logging
import traceback
import app_bedrock

AWS_REGION = os.environ["AWS_REGION"]

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

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
                    extra_instructions = "If you don't know the answer or not in the provided context, just say that you don't know, don't try to make up an answer. Do not answer any question that cannot be answered by the provided context, just say it is not in the provided context. Keep the answer simple."

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
