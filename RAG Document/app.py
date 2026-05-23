import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv
load_dotenv()

## load groq api key

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY") # type: ignore
os.environ['OPENAI_API_KEY']  = os.getenv("OPENAI_API_KEY") # type: ignore


llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
 
    """
    Answer the question based on the context only.
    Please provied the most accurate response based on question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./research_papers")  ## data ingestion step (loading data)
        st.session_state.docs=st.session_state.loader.load()  ## doc loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## text splitter
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


user_prompt = st.text_input("Enter your query from research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vectore db is ready")

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please click 'Document Embedding' first to create vector database")
    else:
        documen_chain=create_stuff_documents_chain(llm,prompt)
        retriever= st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,documen_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input':user_prompt})
        print(f"Response time :{time.process_time()-start}")

        st.write(response['answer'])

        ## with a streamlit expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')