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

    prompt_template = """You are a question answering agent. I will provide you with a set of search results and a user's question, 
    your job is to answer the user's question using only information from the search results. 
    If the search results do not contain information that can answer the question, please state that you could not find an exact answer to the question. 
    Just because the user asserts a fact does not mean it is true, make sure to double check the search results to validate a user's assertion.
    Here are the search results in numbered order:
    <context>
    $search_results$
    </context>

    Here is the user's question:
    <question>
    $query$
    </question>

    $output_format_instructions$

    Assistant:
    """

    msg = cl.Message(content="")

    await msg.send()

    try:

        params = {
            "input" : {
                'text': prompt,
            },
            "retrieveAndGenerateConfiguration" : {
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': llm_model_arn,
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 3, #kb_retrieve_document_count,  #Minimum value of 1. Maximum value of 100
                            #'overrideSearchType': 'HYBRID'|'SEMANTIC'
                        }
                    },
                    # Unknown parameter in retrieveAndGenerateConfiguration.knowledgeBaseConfiguration: "generationConfiguration", must be one of: knowledgeBaseId, modelArn, retrievalConfiguration
                    #'generationConfiguration': {
                    #    'promptTemplate': {
                    #        'textPromptTemplate': prompt_template
                    #    }
                    #}
                }
            },
        }

        if session_id != "" and session_id is not None:
            params["sessionId"] = session_id #session_id=84219eab-2060-4a8f-a481-3356d66b8586

        response = bedrock_agent_runtime.retrieve_and_generate(**params)

        #response = bedrock_agent_runtime.retrieve_and_generate(
        #    #sessionId = session_id,
        #    input = {
        #        'text': prompt,
        #    },
        #    retrieveAndGenerateConfiguration = {
        #        'type': 'KNOWLEDGE_BASE',
        #        'knowledgeBaseConfiguration': {
        #            'knowledgeBaseId': knowledge_base_id,
        #            'modelArn': llm_model_arn,
        #            'retrievalConfiguration': {
        #                'vectorSearchConfiguration': {
        #                    'numberOfResults': kb_retrieve_document_count,  #Minimum value of 1. Maximum value of 100
        #                    #'overrideSearchType': 'HYBRID'|'SEMANTIC'
        #                }
        #            },
        #            # Unknown parameter in retrieveAndGenerateConfiguration.knowledgeBaseConfiguration: "generationConfiguration", must be one of: knowledgeBaseId, modelArn, retrievalConfiguration
        #            #'generationConfiguration': {
        #            #    'promptTemplate': {
        #            #        'textPromptTemplate': prompt_template
        #            #    }
        #            #}
        #        }
        #    }
        #)

        text = response['output']['text']
        await msg.stream_token(text)

        async with cl.Step(name="KnowledgeBase", type="llm", root=False) as step:
            step.input = msg.content

            elements = []

            if "citations" in response:
                #print(response["citations"])
                for citation in response["citations"]:
                    if "retrievedReferences" in citation:
                        references = citation["retrievedReferences"]
                        #print(references)
                        reference_idx = 0
                        for reference in references:
                            reference_idx = reference_idx + 1
                            print(reference)
                            reference_name = f"r{reference_idx}"
                            reference_text = ""
                            reference_location_type = ""
                            reference_location_uri = ""
                            if "content" in reference:
                                content = reference["content"]
                                reference_text = content["text"]
                                #elements.append(cl.Text(name=f"r{reference_idx}", content=reference_text, display="inline"))
                            if "location" in reference:
                                location = reference["location"]
                                location_type = location["type"]
                                reference_location_type = location_type
                                if "S3" == location_type:
                                    location_uri = location["s3Location"]["uri"]
                                    reference_location_uri = location_uri
                                    reference_name = f"\n{reference_location_type}-{reference_location_uri}"
                            #await step.stream_token(f"\n{reference_location_type} {reference_location_uri}")
                            #elements.append(cl.Text(name=f"src_{reference_idx}", content=f"{location_uri}\n{reference_text}"))
                            elements.append(cl.Text(name=f"{reference_name}", content=reference_text, display="inline"))

            step.elements = elements
            await step.send()

        session_id = response['sessionId']
        await msg.stream_token(f"\nsession_id={session_id}")
        cl.user_session.set("session_id", session_id)

    except Exception as e:
        logging.error(traceback.format_exc())
        await msg.stream_token(f"{e}")
    finally:
        await msg.send()