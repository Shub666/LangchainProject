import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
import certifi

load_dotenv()

## Fix SSL certificate issue on Windows
os.environ["SSL_CERT_FILE"] = certifi.where()

## Langsmith tracking

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A chat bot with OpenaAI"

## Prompt template

prompt = ChatPromptTemplate.from_messages([

    ("system","you are a helpful assistant. Please respon to the user question"),
    ("user","Question:{question}")
])

def generateRes(question,api_key,llm,temperature,max_tokens):
    llm=ChatOpenAI(model=llm, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain=prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer


##Tile of the app

st.title("Q&A chat bot with OpenaAI")
api_key = st.sidebar.text_input("Enter your api key:",type="password")
llm=st.sidebar.selectbox("Select OpenAI model",["gpt-4o-mini","gpt-4","gpt-3.5-turbo"])
temperature=st.sidebar.slider("Temperature", min_value=0.0,max_value=1.0,value=0.7)
max_token= st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main interface for user input

st.write("Ask any question")
user_input=st.text_input("You:")

if user_input:
    response = generateRes(user_input,api_key,llm,temperature,max_token)
    st.write(response)
else:
    st.write("Please ask any question")
