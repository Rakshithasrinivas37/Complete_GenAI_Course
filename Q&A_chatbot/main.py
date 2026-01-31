import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import groq

import os
from dotenv import load_dotenv

load_dotenv()

# ## Langsmith tracing
# os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
# os.environ['LANGSMITH_TRAKING'] = "true"
# os.environ['LANGCHAIN_PROJECT'] = os.getenv('Q&A Chatbot with Groq')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your a helpful AI assistant. Please response to user queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, model, temperature, max_tokens):
    model = Ollama(model=model)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    print("Invoking.....")
    response = chain.invoke({'question': question})
    print("Generated response")
    return response

## Title of web app
st.title("Q&A chatbot with Groq")

##Dropdown to select the various models
model = st.sidebar.selectbox("Select the model", ['llama3.2:1b'])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value=50, max_value=300, value=150)

st.write("Type your question")
user_input = st.text_input("You: ")

if user_input:
    print("Calling generate response.....")
    response = generate_response(user_input, model, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user query")