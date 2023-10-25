#importing libraries.
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import chainlit as cl

# setting up the key
from key import HuggingFace_key
HuggingFace_API = HuggingFace_key
os.environ['HuggingFace_API_TOKEN']= HuggingFace_API

# setting up the chatbot model
model_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token = os.environ['HuggingFace_API_TOKEN'],
                    repo_id = model_id,
                    model_kwargs={"temperature":0.82, "max_new_tokens":1000})

# template telling bot how to behave.
template = """You might be a AI assistant that suggest solution to user queries.

{query}.
"""

# chainlit to create UI

# decorator from Chainlit for langchain
@cl.langchain_factory(use_async=False)
# incorporated the LLM code.
def manufacturing_facility(query):
    prompt = PromptTemplate(template=template, input_variables=['query'])
    chat_chain = LLMChain(llm=llm,
                        prompt=prompt,
                          input_variables={'query':query},
                        verbose=True)
    return chat_chain