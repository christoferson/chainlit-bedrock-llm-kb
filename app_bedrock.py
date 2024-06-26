import chainlit as cl
import json

class BedrockModelStrategy():

    def create_prompt(self, application_options: dict, context_info: str, query: str) -> str:

        prompt = ""
        if "" == context_info:
            prompt = self._create_prompt(application_options, query)
        else:
            prompt = self._create_prompt_with_context(application_options, context_info, query)

        return prompt

    def _create_prompt(self, application_options: dict, query: str) -> str:

        prompt = f"""Please answer the following question to the best of your ability. If you are unsure or don't have enough information to provide a confident answer, simply say "I don't know" or "I'm not sure."

        \n\nHuman: {query}

        Assistant:
        """

        return prompt

    def _create_prompt_with_context(self, application_options: dict, context_info: str, query: str) -> str:

        option_terse = application_options.get("option_terse")
        option_strict = application_options.get("option_strict")

        strict_instructions = ""
        if option_strict == True:
            strict_instructions = "Only answer if you know the answer with certainty and is evident from the provided context. Otherwise, just say that you don't know and don't make up an answer."

        terse_instructions = ""
        if option_terse == True:
            terse_instructions = "Unless otherwise instructed, omit any preamble and provide terse and concise one liner answer."

        prompt = f"""Please answer the question with the provided context while following instructions provided. 
        {terse_instructions} {strict_instructions}
        Here is the context: {context_info}
        \n\nHuman: {query}

        Assistant:
        """

        return prompt

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        pass

    def send_request(self, request:dict, bedrock_runtime, bedrock_model_id:str):
        response = bedrock_runtime.invoke_model_with_response_stream(modelId = bedrock_model_id, body = json.dumps(request))
        return response

    async def process_response(self, response, msg : cl.Message):
        stream = response["body"]
        await self.process_response_stream(stream, msg)

    async def process_response_stream(self, stream, msg : cl.Message):
        print("unknown")
        await msg.stream_token("unknown")

    #Todo Utils: amazon-bedrock-invocationMetrics 


class BedrockModelStrategyFactory():

    @staticmethod
    def create(bedrock_model_id : str) -> BedrockModelStrategy:

        model_strategy = None

        provider = bedrock_model_id.split(".")[0]
        
        if bedrock_model_id.startswith("anthropic.claude-3"): #"anthropic.claude-3-sonnet-20240229-v1:0": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
            #model_strategy = AnthropicClaude3MsgBedrockModelStrategy()
            model_strategy = AnthropicClaude3MsgBedrockModelAsyncStrategy()
        elif provider == "anthropic": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
            model_strategy = AnthropicBedrockModelStrategy()
        elif provider == "ai21": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jurassic2.html
            model_strategy = AI21BedrockModelStrategy()
        elif provider == "cohere": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
            model_strategy = CohereBedrockModelStrategy()
        elif provider == "amazon": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
            model_strategy = TitanBedrockModelStrategy()
        elif provider == "meta": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
            model_strategy = MetaBedrockModelStrategy()
        elif provider == "mistral": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
            model_strategy = MistralBedrockModelStrategy()
        else:
            raise ValueError(f"Not Supported Model. Model={bedrock_model_id} Provide={provider}")
        
        return model_strategy
    
class AnthropicBedrockModelStrategy(BedrockModelStrategy):

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "prompt": prompt,
            "temperature": inference_parameters.get("temperature"),
            "top_p": inference_parameters.get("top_p"), #0.5,
            "top_k": inference_parameters.get("top_k"), #300,
            "max_tokens_to_sample": inference_parameters.get("max_tokens_to_sample"), #2048,
            #"stop_sequences": []
        }
        return request

    async def process_response_stream(self, stream, msg : cl.Message):
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    object = json.loads(chunk.get("bytes").decode())
                    #print(object)
                    if "completion" in object:
                        completion = object["completion"]
                        #print(completion)
                        await msg.stream_token(completion)
                    stop_reason = None
                    if "stop_reason" in object:
                        stop_reason = object["stop_reason"]
                    
                    if stop_reason == 'stop_sequence':
                        invocation_metrics = object["amazon-bedrock-invocationMetrics"]
                        if invocation_metrics:
                            input_token_count = invocation_metrics["inputTokenCount"]
                            output_token_count = invocation_metrics["outputTokenCount"]
                            latency = invocation_metrics["invocationLatency"]
                            lag = invocation_metrics["firstByteLatency"]
                            stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                            await msg.stream_token(f"\n\n{stats}")

class AnthropicClaude3MsgBedrockModelStrategy(BedrockModelStrategy):

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:

        user_message =  {"role": "user", "content": f"{prompt}"}
        messages = [user_message]

        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": inference_parameters.get("temperature"),
            "top_p": inference_parameters.get("top_p"), #0.5,
            "top_k": inference_parameters.get("top_k"), #300,
            "max_tokens": inference_parameters.get("max_tokens_to_sample"), #2048,
            #"system": system_prompt,
            "messages": messages
            #"stop_sequences": []
        }
        return request

    def send_request(self, request:dict, bedrock_runtime, bedrock_model_id:str):
        response = bedrock_runtime.invoke_model(modelId = bedrock_model_id, body = json.dumps(request))
        print(response)
        return response

    async def process_response(self, response, msg : cl.Message):
        response_body = json.loads(response.get('body').read())
        print(response_body)
        contents = response_body["content"]
        for content in contents:
            await msg.stream_token(f"{content['text']}")
        usage = response_body["usage"]
        await msg.stream_token(f"token.in={usage['input_tokens']},token.out={usage['output_tokens']}")

    async def process_response_stream(self, stream, msg : cl.Message):
        pass


class AnthropicClaude3MsgBedrockModelAsyncStrategy(BedrockModelStrategy):

    def create_prompt(self, application_options: dict, context_info: str, query: str) -> str:

        option_terse = application_options.get("option_terse")
        option_strict = application_options.get("option_strict")
        option_align_units_of_measure = application_options.get("option_align_units_of_measure")
        option_align_geographic_divisions = application_options.get("option_align_geographic_divisions")
        option_state_conclusions_first = application_options.get("option_state_conclusions_first")
        option_source_table_markdown_display = application_options.get("option_source_table_markdown_display")

        strict_instructions = ""
        if option_strict == True:
            strict_instructions = "- Only answer if you know the answer with certainty and is evident from the provided context."

        terse_instructions = ""
        if option_terse == True:
            terse_instructions = "Unless otherwise instructed, omit any preamble and provide terse and concise one liner answer."

        align_units_of_measure_instruction = ""
        if option_align_units_of_measure == True:
            align_units_of_measure_instruction = "- If the unit of measure of the question does not match that of the source document, convert the input to the same unit of measure as the source first before deciding."

        state_conclusions_first_instruction = ""
        if option_state_conclusions_first == True:
            state_conclusions_first_instruction = "- Always provide the conclusion or answer first then state the reason as a separate paragraph."

        source_table_markdown_display_instructions = ""
        if option_source_table_markdown_display == True:
            source_table_markdown_display_instructions = "- Present source data in tabular form as markdown."
        
        align_geographic_divisions_instructions = ""
        if option_align_geographic_divisions == True:
            align_geographic_divisions_instructions = "- When encountering geographic locations in user questions, first establish the administrative division specified in the source document. If the input location differs from this specified division (e.g., the source document uses prefectures or provinces while the user mentions cities), determine the corresponding administrative division (e.g., prefecture or province) where the mentioned city is situated. Use this determined administrative division to answer the question in accordance with the criteria outlined in the source document."


        prompt = f"""Please answer the question with the provided context while following instructions provided. {terse_instructions}
        <context>{context_info}</context>
        <instructions>
        - Do not reformat, or convert any numeric values. Inserting commas is allowed for readability. {align_units_of_measure_instruction} {align_geographic_divisions_instructions} {state_conclusions_first_instruction} {source_table_markdown_display_instructions} {strict_instructions}
        </instructions>
        <question>{query}</question>"""

        return prompt
    
    def create_request(self, inference_parameters: dict, prompt : str) -> dict:

        user_message =  {"role": "user", "content": f"{prompt}"}
        messages = [user_message]

        request = {
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": inference_parameters.get("temperature"),
            "top_p": inference_parameters.get("top_p"), #0.5,
            "top_k": inference_parameters.get("top_k"), #300,
            "max_tokens": inference_parameters.get("max_tokens_to_sample"), #2048,
            "system": inference_parameters.get("system_message") if inference_parameters.get("system_message") else  "You are a helpful assistant.",
            "messages": messages
            #"stop_sequences": []
        }
        return request

    def send_request(self, request:dict, bedrock_runtime, bedrock_model_id:str):
        response = bedrock_runtime.invoke_model_with_response_stream(modelId = bedrock_model_id, body = json.dumps(request))
        print(response)
        return response

    async def process_response_stream(self, stream, msg : cl.Message):

        for event in stream:
            if event["chunk"]:
                chunk = json.loads(event["chunk"]["bytes"])

                if chunk['type'] == 'message_start':
                    #print(f"Input Tokens: {chunk['message']['usage']['input_tokens']}")
                    pass

                elif chunk['type'] == 'message_delta':
                    #print(f"\nStop reason: {chunk['delta']['stop_reason']}")
                    #print(f"Stop sequence: {chunk['delta']['stop_sequence']}")
                    #print(f"Output tokens: {chunk['usage']['output_tokens']}")
                    pass

                elif chunk['type'] == 'content_block_delta':
                    if chunk['delta']['type'] == 'text_delta':
                        text = chunk['delta']['text']
                        await msg.stream_token(f"{text}")

                elif chunk['type'] == 'message_stop':
                    invocation_metrics = chunk['amazon-bedrock-invocationMetrics']
                    input_token_count = invocation_metrics["inputTokenCount"]
                    output_token_count = invocation_metrics["outputTokenCount"]
                    latency = invocation_metrics["invocationLatency"]
                    lag = invocation_metrics["firstByteLatency"]
                    stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                    await msg.stream_token(f"\n\n{stats}")

            elif event["internalServerException"]:
                exception = event["internalServerException"]
                await msg.stream_token(f"\n\n{exception}")
            elif event["modelStreamErrorException"]:
                exception = event["modelStreamErrorException"]
                await msg.stream_token(f"\n\n{exception}")
            elif event["modelTimeoutException"]:
                exception = event["modelTimeoutException"]
                await msg.stream_token(f"\n\n{exception}")
            elif event["throttlingException"]:
                exception = event["throttlingException"]
                await msg.stream_token(f"\n\n{exception}")
            elif event["validationException"]:
                exception = event["validationException"]
                await msg.stream_token(f"\n\n{exception}")
            else:
                await msg.stream_token(f"Unknown Stream Token: {event}")

class CohereBedrockModelStrategy(BedrockModelStrategy):

    def create_prompt(self, application_options: dict, context_info: str, query: str) -> str:

        option_terse = application_options.get("option_terse")
        option_strict = application_options.get("option_strict")

        strict_instructions = ""
        if option_strict == True:
            strict_instructions = "Only answer if you know the answer with certainty and is evident from the provided context. Otherwise, just say that you don't know and don't make up an answer."

        terse_instructions = ""
        if option_terse == True:
            terse_instructions = "Unless otherwise instructed, omit any preamble and provide terse and concise one liner answer."

        prompt = f"""Please answer the question with the provided context while following instructions provided. 
        {terse_instructions} {strict_instructions}
        Here is the context: {context_info}
        \n\nHuman: {query}

        AI:"""

        return prompt

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "prompt": prompt,
            "temperature": inference_parameters.get("temperature"),
            "p": inference_parameters.get("top_p"), #0.5,
            "k": inference_parameters.get("top_k"), #300,
            "max_tokens": inference_parameters.get("max_tokens_to_sample"), #2048,
            "stream": True,
            #"stop_sequences": []
        }
        return request

    async def process_response_stream(self, stream, msg : cl.Message):
        #print("cohere")
        #await msg.stream_token("Cohere")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    object = json.loads(chunk.get("bytes").decode())
                    is_finished = object["is_finished"]
                    if is_finished == False:
                        await msg.stream_token(object["text"])
                    else:
                        if "amazon-bedrock-invocationMetrics" in object:
                            invocation_metrics = object["amazon-bedrock-invocationMetrics"]
                            if invocation_metrics:
                                input_token_count = invocation_metrics["inputTokenCount"]
                                output_token_count = invocation_metrics["outputTokenCount"]
                                latency = invocation_metrics["invocationLatency"]
                                lag = invocation_metrics["firstByteLatency"]
                                stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                                await msg.stream_token(f"\n\n{stats}")


class TitanBedrockModelStrategy(BedrockModelStrategy):

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "inputText": prompt,
            "textGenerationConfig": {
                "temperature": inference_parameters.get("temperature"),
                "topP": inference_parameters.get("top_p"), #0.5,
                #"top_k": inference_parameters.get("top_k"), #300,
                "maxTokenCount": inference_parameters.get("max_tokens_to_sample"), #2048,
                #"stop_sequences": []
            }
        }
        return request

    async def process_response_stream(self, stream, msg : cl.Message):
        #print("titan")
        #await msg.stream_token("Titan")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    object = json.loads(chunk.get("bytes").decode())
                    #print(object)
                    if "outputText" in object:
                        completion = object["outputText"]
                        await msg.stream_token(completion)
                    if "completionReason" in object:
                        finish_reason = object["completionReason"]
                        if finish_reason:
                            if "amazon-bedrock-invocationMetrics" in object:
                                invocation_metrics = object["amazon-bedrock-invocationMetrics"]
                                if invocation_metrics:
                                    input_token_count = invocation_metrics["inputTokenCount"]
                                    output_token_count = invocation_metrics["outputTokenCount"]
                                    latency = invocation_metrics["invocationLatency"]
                                    lag = invocation_metrics["firstByteLatency"]
                                    stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag} finish_reason={finish_reason}"
                                    await msg.stream_token(f"\n\n{stats}")


# Issue - Infinite Please answer the question with the provided context while following instructions provided.
class MetaBedrockModelStrategy(BedrockModelStrategy):

    def create_prompt(self, application_options: dict, context_info: str, query: str) -> str:

        option_terse = application_options.get("option_terse")
        option_strict = application_options.get("option_strict")

        strict_instructions = ""
        if option_strict == True:
            strict_instructions = "Only answer if you know the answer with certainty and is evident from the provided context. Otherwise, just say that you don't know and don't make up an answer."

        terse_instructions = ""
        if option_terse == True:
            terse_instructions = "Unless otherwise instructed, omit any preamble and provide terse and concise one liner answer."

        prompt = f"""<<SYS>>Please answer the question with the provided context while following instructions provided. 
        {terse_instructions} {strict_instructions}<</SYS>>
        Here is the context: {context_info}
        
        Human: {query}

        Assistant:"""

        return prompt
    
    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "prompt": prompt,           
            "temperature": inference_parameters.get("temperature"),
            "top_p": inference_parameters.get("top_p"), #0.5,
            #"top_k": inference_parameters.get("top_k"), #300,
            "max_gen_len": inference_parameters.get("max_tokens_to_sample"), #2048,
            #"stop_sequences": []
        }
        return request

    async def process_response_stream(self, stream, msg : cl.Message):
        print("meta")
        await msg.stream_token("Meta")
        if stream:
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    object = json.loads(chunk.get("bytes").decode())
                    print(object)
                    if "generation" in object:
                        completion = object["generation"]
                        await msg.stream_token(completion)
                    if "stop_reason" in object:
                        finish_reason = object["stop_reason"]
                        if finish_reason:
                            if "amazon-bedrock-invocationMetrics" in object:
                                invocation_metrics = object["amazon-bedrock-invocationMetrics"]
                                if invocation_metrics:
                                    input_token_count = invocation_metrics["inputTokenCount"]
                                    output_token_count = invocation_metrics["outputTokenCount"]
                                    latency = invocation_metrics["invocationLatency"]
                                    lag = invocation_metrics["firstByteLatency"]
                                    stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag} finish_reason={finish_reason}"
                                    await msg.stream_token(f"\n\n{stats}")


class AI21BedrockModelStrategy(BedrockModelStrategy):

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "prompt": prompt,           
            "temperature": inference_parameters.get("temperature"),
            "topP": inference_parameters.get("top_p"), #0.5,
            #"top_k": inference_parameters.get("top_k"), #300,
            "maxTokens": inference_parameters.get("max_tokens_to_sample"), #2048,
            #"stop_sequences": []
        }
        return request

    def send_request(self, request:dict, bedrock_runtime, bedrock_model_id:str):
        response = bedrock_runtime.invoke_model(modelId = bedrock_model_id, body = json.dumps(request))
        return response
    
    async def process_response_stream(self, stream, msg : cl.Message):
        #await msg.stream_token(f"AI21")
        
        object = json.loads(stream.read())
        #print(json.dumps(object, indent=2))
        #print(object.get('completions')[0].get('data').get('text'))
        text = object.get('completions')[0].get('data').get('text')
        await msg.stream_token(f"{text}\n")

class MistralBedrockModelStrategy(BedrockModelStrategy):

    def create_request(self, inference_parameters: dict, prompt : str) -> dict:
        request = {
            "prompt": prompt,
            "temperature": inference_parameters.get("temperature"),
            "top_p": inference_parameters.get("top_p"), #0.5,
            "top_k": inference_parameters.get("top_k"), #300,
            "max_tokens": inference_parameters.get("max_tokens_to_sample"), #2048,
            #"stop_sequences": []
        }
        return request

    async def process_response_stream(self, stream, msg : cl.Message):
        if stream:
            for event in stream:
                #print(f"Event: {event}")
                chunk = event.get("chunk")
                if chunk:
                    object = json.loads(chunk.get("bytes").decode())
                    #print(object)
                    if "outputs" in object:
                        outputs = object["outputs"]
                        for index, output in enumerate(outputs):
                            text = output["text"]
                            if text == "\n":
                                #print(f"{len(text)} '{text}'")
                                pass
                                #await msg.stream_token(text)
                            else:
                                await msg.stream_token(text)

                            stop_reason = None
                            if "stop_reason" in output:
                                stop_reason = output["stop_reason"]
                            
                            if stop_reason == 'stop' or stop_reason == 'length':
                                invocation_metrics = object["amazon-bedrock-invocationMetrics"]
                                if invocation_metrics:
                                    input_token_count = invocation_metrics["inputTokenCount"]
                                    output_token_count = invocation_metrics["outputTokenCount"]
                                    latency = invocation_metrics["invocationLatency"]
                                    lag = invocation_metrics["firstByteLatency"]
                                    stats = f"token.in={input_token_count} token.out={output_token_count} latency={latency} lag={lag}"
                                    await msg.stream_token(f"\n\n{stats}")
