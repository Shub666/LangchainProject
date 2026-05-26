import validators,streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
import os


## streamlit app

st.set_page_config(page_title="Langchain: Summarize content from Youtube or Website")
st.title("Langchain: Summarize content from Youtube or Website")
st.subheader("Summarize URL")


with st.sidebar:
    g_api_key=st.text_input("Groq API Key", value="", type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

llm= ChatGroq(api_key=g_api_key,model="openai/gpt-oss-120b") # type: ignore

prompt_template="""
Provide a summery of the following content in 300 words:
Content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize content from Youtube or Website"):
    if not g_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("Please enter valid URL. IT can be youtube or website")
    else:
        try:
            with st.spinner("waiting..."):
                ## load data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(youtube_url=generic_url,add_video_info=False)
                else:
                    loader =UnstructuredURLLoader(urls=[generic_url])
                docs=loader.load()

                ## chain for summarization
                chain=load_summarize_chain(llm=llm,chain_type="stuff",prompt=prompt)
                output_summery=chain.run(docs)

                st.success(output_summery)
        except Exception as e:
            st.exception(e,width="stretch")

