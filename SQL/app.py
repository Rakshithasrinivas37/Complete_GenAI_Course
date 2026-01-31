import streamlit as st
from pathlib import Path

from langchain_classic.agents import create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine

import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="Langchain: Chat with SQL DB")
st.title("Chatbot with SQL Database")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite3 Database - student.db", "Connect to MYSQL Database"]

selected_opt = st.sidebar.radio(label="Chhose the DB", options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MYSQL Host")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password = st.sidebar.text_input("MYSQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label = "GROQ API Key", type="password")

if not db_uri:
    st.info("Please enter the database info and uri")

if not api_key:
    st.info("Please provide GROQ API Key")

## Model Creation
model = ChatGroq(model="qwen/qwen3-32b", groq_api_key=api_key)

@st.cache_resource(ttl='2h')
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_filepath = (Path(__file__).parent/"student.db").absolute()
        print(db_filepath)

        creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    
if db_uri==MYSQL:
    db = configure_db(
        db_uri=db_uri,
        mysql_host=mysql_host,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        mysql_db=mysql_db
    )
else:
    db = configure_db(db_uri)

## create toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=model)

agent = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear Message history"):
    st.session_state["messages"] = [{'role': 'assistant', 'content': 'How can I help you?'}]
        
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

user_input = st.chat_input(placeholder="Ask anything from the Database")

if user_input:
    st.session_state.messages.append({'role': 'user', "content": user_input})
    st.chat_message('user').write(user_input)

    with st.chat_message('assistant'):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(input=st.session_state.messages, callbacks=[streamlit_callback], handle_parsing_error=True)
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)