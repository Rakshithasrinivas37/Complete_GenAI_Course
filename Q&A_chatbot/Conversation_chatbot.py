## RAG Q&A conversation chatbot with pdf including chat history

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']="Your_API_KEY"
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up Streamlit 
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

## Input the Groq API Key
api_key=st.text_input("Enter your Groq API key:",type="password")

## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="qwen/qwen3-32b")

    ## chat interface

    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        ## Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=splits, embeddings=embeddings)
        retriever = vector_store.as_retriever()

        question_context_system_prompt = (
            "Given the prior conversation and the user's latest question, "
            "determine whether the question depends on earlier context. "
            "If it does, rewrite it as a standalone question that preserves the original meaning. "
            "Do not provide an answer. If the question is already standalone, return it as-is."
        )


        question_context_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", question_context_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(model, retriever, question_context_prompt)

        ## Question answer prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}")
            ]
        )

        qa_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()

            return st.session_state.store[session_id]
        
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_message_key="chathistory",
            output_message_key="Answer"
        )

        user_input = st.text_input("Type your question...")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )

            st.write(st.session_state.store)
            st.success("Assistant: ", response['answer'])
            st.write("Chat history: ", session_history.messages)

else:
    st.warning("Please enter your GROQ API KEY")





                    