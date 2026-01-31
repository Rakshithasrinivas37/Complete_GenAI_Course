import validators
import streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Setting up streamlit app
st.set_page_config(page_title="Langchain: Summrize Text from Youtube or Website")
st.title("Langchain: Summrize Text from Youtube or Website")
st.subheader('Summarize URL')

## Get the Groq API key and url field to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type='password')

url = st.text_input("URL", label_visibility="collapsed")

model = ChatGroq(model="qwen/qwen3-32b", groq_api_key=groq_api_key)

prompt = '''
    Summarize the following content in 500 words.
    Content; {text}
'''

prompt_template = PromptTemplate(template=prompt, input_variables = ['text'])

if st.button("Summarize"):
    ## Valiadte the input parameters
    if not groq_api_key.strip() or not url.strip():
        st.error("Please fill all the fields")
    elif not validators.url(url):
        st.error("Please provide valid URL to summarize")
    else:
        try:
            with st.spinner("waiting..."):
                ## Load the website data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                data = loader.load()

                ## Create chain summarization
                chain = load_summarize_chain(model, chain_type="stuff", prompt=prompt_template)
                output_summary = chain.invoke(data)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
