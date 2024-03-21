import chainlit as cl
import json

class BedrockModelStrategy():

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


class BedrockModelStrategyFactory():

    @staticmethod
    def create(bedrock_model_id : str) -> BedrockModelStrategy:

        model_strategy = None

        provider = bedrock_model_id.split(".")[0]
        
        if bedrock_model_id.startswith("anthropic.claude-3"): #"anthropic.claude-3-sonnet-20240229-v1:0": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
            model_strategy = AnthropicClaude3MsgBedrockModelStrategy()
        elif provider == "anthropic": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
            model_strategy = AnthropicBedrockModelStrategy()
        #elif provider == "ai21": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jurassic2.html
        #    model_strategy = AI21BedrockModelStrategy()
        #elif provider == "cohere": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
        #    model_strategy = CohereBedrockModelStrategy()
        #elif provider == "amazon": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
        #    model_strategy = TitanBedrockModelStrategy()
        #elif provider == "meta": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
        #    model_strategy = MetaBedrockModelStrategy()
        #elif provider == "mistral": # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
        #    model_strategy = MistralBedrockModelStrategy()
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
            #"prompt": prompt,
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
