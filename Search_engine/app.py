import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq

import os

## Search Tools
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

search = DuckDuckGoSearchRun(name="search")

st.title("Langchain - chat with search")

## Sidebar for settings
st.sidebar.title("settings")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {'role': "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({'role':'user', "content": prompt})
    st.chat_message("user").write(prompt)

    model = ChatGroq(model="qwen/qwen3-32b", groq_api_key=groq_api_key)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, model, AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.invoke(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":'assistant', "content": response})
        st.write(response)