import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain, LLMMathChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool, initialize_agent
from langchain_community.callbacks import StreamlitCallbackHandler
import numexpr as ne


## Setup streamlit application
st.set_page_config(page_title="Math Problem Solver and Data Search Assistant")

st.title("Math Problem Solver")

groq_api_key = st.sidebar.text_input(label="GROQ API Key", type="password")

if not groq_api_key:
    st.info("Please provide your GROQ API Key")
    st.stop()

model = ChatGroq(model="qwen/qwen3-32b", groq_api_key=groq_api_key)

## Initialize the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find the various information that required to solve math problems"
)

## Initialize the Math tool
def calculator(expr: str):
    try:
        return str(ne.evaluate(expr))
    except Exception as e:
        return f"Invalid math expression: {e}"
    
calculator = Tool(
    name="Calculator",
    func=calculator,
    description="A tool for solving the math related problems. An input can only be mathematical expressions"
)

# Reasoning tool
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a math tutor. Solve the following problem step by step.

Rules:
1. Number each step clearly.
2. Explain what you are doing in each step in simple words.
3. If any calculation is needed, show the calculation.
4. Give the final answer at the end.
5. Do NOT just give the final number.

Problem:
{question}

Step-by-step solution:
"""
)

## Create chain
chain = LLMChain(llm=model, prompt=prompt)

reasoning_tool = Tool(
    name='reasoning',
    func=chain.run,
    description="Always explain math problems step-by-step, showing reasoning and calculations."
)

##Initialize the agents
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant",
         "content":"Hi, I'm math chatbot who can answer all your math related questions and explain each steps"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

##Generate response 
def generate_response(question, st_cb):
    return assistant_agent.invoke({'input': question}, callbacks=[st_cb])

## Start conversation
question = st.text_area("Type your question")
if st.button("Solve the Problem"):
    if question:
        with st.spinner("Generating response......."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = generate_response(question, st_cb)

            st.session_state.messages.append({'role': 'assistant', 'content': response['output']})
            st.write("Response: \n")
            st.success(response['output'])
    else:
        st.warning("Please enter your input")