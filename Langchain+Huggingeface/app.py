import validators
import streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

## Setting up streamlit app
st.set_page_config(page_title="Langchain: Summarize Text from Youtube or Website")
st.title("Langchain: Summarize Text from Youtube or Website")
st.subheader('Summarize URL')

## Get the Groq API key and url field to be summarized
with st.sidebar:
    hf_api_key = st.text_input("HF API Key", value="", type='password')

url = st.text_input("URL", label_visibility="collapsed")

llm = HuggingFaceEndpoint(
  repo_id="meta-llama/Llama-3.1-8B-Instruct",
  huggingfacehub_api_token = hf_api_key,
)

chat_model = ChatHuggingFace(llm=llm)

prompt = '''
    Summarize the following content in 500 words.
    Content; {text}
'''

prompt_template = PromptTemplate(template=prompt, input_variables = ['text'])

if st.button("Summarize"):
    ## Valiadte the input parameters
    if not hf_api_key.strip() or not url.strip():
        st.error("Please fill all the fields")
    elif not validators.url(url):
        st.error("Please provide valid URL to summarize")
    else:
        try:
            with st.spinner("waiting..."):
                ## Load the website data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(
                    url,
                    add_video_info=False,
                    language=["en"]
                )
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                data = loader.load()
                print("Number of documents:", len(data))

                ## Create chain summarization
                chain = load_summarize_chain(chat_model, chain_type="stuff", prompt=prompt_template)
                output_summary = chain.invoke(data)

                st.success(output_summary['output_text'])
        except Exception as e:
            print(e)
            st.exception(e)
