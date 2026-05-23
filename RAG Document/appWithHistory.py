import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import time
from dotenv import load_dotenv
load_dotenv()


# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY") # type: ignore
os.environ['HF_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN") # type: ignore

embeding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

## stremlit app
st.title("")
st.write("Upload pdf and chat with content ")

## input groq api key
g_api_key = st.text_input("Enter your Groq api key",type="password")

## check if api key is provided
if g_api_key:
    llm=ChatGroq(api_key=g_api_key,model="llama-3.1-8b-instant") # type: ignore

    session_id=st.text_input("Session ID", value="default")

    ## manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files = st.file_uploader("Ch0ose a pdf file ",type="pdf")

    if uploaded_files:
        documents= []
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_files.getvalue())
                file_name=uploaded_files.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    ## Split and create embedding for the documents
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorestore= Chroma.from_documents(documents=splits, embedding=embeding)
        retriever = vectorestore.as_retriever()

        contextualize_q_system_prompt= (
            "Given  a chat history and the latest user question"
            "which might refrence context in the chat history"
            "formulate a standlone question which can be understood"
            "without the chat history, Do Not answer the question,"
            "just reformulate it if needed and otherwise return it as is."

        )
        contextualize_q_prompt= ChatPromptTemplate.from_messages([

            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ])

        history_aware_retrver=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## answer question 
        systm_prompt=(
            "you are a assistence for question-answer tasks."
            "Use the following pieces of retrived context to answer to the question."
            "if you don't know answer say that you don't know"
            "keep answer concise."
            "\n\n0" \
            "{context}"
        )

        qa_prompt= ChatPromptTemplate.from_messages([

            ("system", systm_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),

        ])

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retrver,question_answer_chain)
        

        def get_session_history(session:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]= ChatMessageHistory()
            return st.session_state.store[session_id]

        
        conversatinal_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chathistory",
            output_messages_key="answer"
        )

        user_input= st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response= conversatinal_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("chat history:",session_history.messages)
else:
    st.warning("Please enter your api key")