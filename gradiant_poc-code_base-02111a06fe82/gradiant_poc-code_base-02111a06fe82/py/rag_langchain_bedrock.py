from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

import boto3
import json

region = "eu-west-3"
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=region,
)

# add any file of choice please
loader = TextLoader('')
documents = loader.load()

#print(documents)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

#print(texts)


from langchain_aws.embeddings import BedrockEmbeddings
embeddings = BedrockEmbeddings(client=bedrock_runtime, model_id="amazon.titan-embed-text-v2:0")#"amazon.titan-embed-text-v1"

#print(embeddings)

db = Chroma.from_documents(texts, embeddings)

query = "explain about the transformers"

retriever = db.similarity_search(query, k=3)
full_context = '\n'.join([f'Document {indexing+1}: ' + i.page_content for indexing, i in enumerate(retriever)])

print('context')
print(full_context)

prompt_template = f"""Answer the user’s question solely only on the information provided between <></> XML tags. Think step by step and provide detailed instructions.
<context>
{full_context}
</context>

Question: {query}
Answer:"""

from langchain.prompts import PromptTemplate
PROMPT = PromptTemplate.from_template(prompt_template)

prompt_data_input = PROMPT.format(human_input=query, context=full_context)

#print(prompt_data_input)

body = json.dumps({"inputText": prompt_data_input, "textGenerationConfig": {"temperature":0,"topP":1,"maxTokenCount":1000}})
accept = "application/json"
contentType = "application/json"

modelId = "amazon.titan-text-express-v1"

response = bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

generated_response_body = json.loads(response.get("body").read())
print('output')
print(generated_response_body.get("results")[0].get("outputText").strip())