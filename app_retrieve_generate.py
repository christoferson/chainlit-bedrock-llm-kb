import os
import boto3
import chainlit as cl
import logging
import traceback

AWS_REGION = os.environ["AWS_REGION"]

bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)

async def main_retrieve_and_generate(message: cl.Message):

    session_id = cl.user_session.get("session_id") 
    knowledge_base_id = cl.user_session.get("knowledge_base_id") 
    llm_model_arn = cl.user_session.get("llm_model_arn") 
    inference_parameters = cl.user_session.get("inference_parameters")
    kb_retrieve_document_count = cl.user_session.get("kb_retrieve_document_count")

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
                            'numberOfResults': kb_retrieve_document_count,  #Minimum value of 1. Maximum value of 100
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