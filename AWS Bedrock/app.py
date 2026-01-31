import json
import os
import sys
import boto3
import streamlit as st

from langchain_classic.embeddings import BedrockEmbeddings
from langchain_classic.llms.bedrock import Bedrock

import numpy as np
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.document_loaders import PyPDFDirectoryLoader

from langchain_community.vectorstores import FAISS

from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import retrieval_qa

## To create bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap=1000) 
    docs = text_splitter.split_documents(documents)
    return docs

## Vector embedding and vector store
def get_vector_store(docs):
    vector_store = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )

    vector_store.save_local("faiss_index")

def get_claude_llm():
    model = Bedrock(model_id = "ai21.j2-mid-v1", client=bedrock,
                    model_kwargs={'maxTokens': 512})
    return model

def get_llama_llm():
    model = Bedrock(model_id = "meta.llama2-70b-chat-v1", client=bedrock,
                    model_kwargs={'max_gen_len': 512})
    return model

prompt = '''
    Human: Use the following pieces of context to provide a concise
    anser to the question at the end but use atleast summarize with 250
    words with detailed explanations. If you dont know the answer,
    Just say that you dont know, dont try to make up an answer
    <context>
    {context}
    <?context>

    Question: {question}

    Assistant:
'''

PROMPT = PromptTemplate(
    template = prompt,
    input_variables=['context', 'question']
)

def get_response_llm(llm, vector_store, query):
    qa = retrieval_qa(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type = "similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={'prompt': PROMPT}
    )

    answer = qa({'query': query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Please type your question")

    with st.sidebar:
        st.title("Update or Create Vector store")

        if st.button("Vector update"):
            with st.spinner("Processing...."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing...."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)

            model = get_claude_llm()

            st.write(get_response_llm(model, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama Output"):
        with st.spinner("Processing...."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)

            model = get_llama_llm()

            st.write(get_response_llm(model, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()