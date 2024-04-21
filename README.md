# chainlit-bedrock-llm-kb
chainlit-bedrock-llm-kb

##### Install Dependencies

pip install --upgrade boto3
pip install --upgrade chainlit
pip install --upgrade python-dotenv


##### Launch Locally

chainlit run app.py -h

##### Prompt Guides (Claude)

Skip the preamble and provide concise answers.

Role Prompting (Improve Accuracy, Change Tone and Demeanor)

Use XML

When dealing with long documents, ask your question at the bottom

Think step by step. - For complex tasks

Before answering, please think about the question within <thinking></thinking> XML Tags. Then, answer the question within <answer></answer> XML Tags.

##### Links

- https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_ListKnowledgeBases.html

- https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/list_knowledge_bases.html#

- https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve_and_generate.html

- https://docs.google.com/presentation/d/1tjvAebcEyR8la3EmVwvjC7PHR8gfSrcsGKfTPAaManw/edit?pli=1#slide=id.g297e9aa6f0f_0_2411

- https://docs.google.com/spreadsheets/d/19jzLgRruG9kjUQNKtCg1ZjdD6l6weA6qRXG5zLIAhC8/edit#gid=352404979

- https://docs.anthropic.com/claude/docs/prompt-engineering

- https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html