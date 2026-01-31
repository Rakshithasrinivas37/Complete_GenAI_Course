import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

##load the Groq API
groq_api_key = "Your_API_KEY"

model = ChatGroq(model="qwen/qwen3-32b", groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
'''
)

def create_vector_embedding():
    ##Session state variable: It will be able to access in other functions, whenever it is rquired.
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader('pdfs')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        print("Docs:", len(st.session_state.final_documents))
        print("Embeddings:", st.session_state.embeddings)

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    return

user_prompt = st.text_input("Enter youe query from the pdf")

if st.button("Document embedding"):
    create_vector_embedding()
    st.write("Vector database is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"response time: {time.process_time() - start}")

    st.write(response['answer'])

    ## with a streamlit expander
    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------------")