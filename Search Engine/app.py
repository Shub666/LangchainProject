import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv


## tools

api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250) # type: ignore
wiki= WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_arxiv_wiki=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250) # type: ignore
arxiv= ArxivQueryRun(api_wrapper=api_arxiv_wiki)

search= DuckDuckGoSearchRun(name="Search")

st.title("Chat with Search")

st.sidebar.title("Setting")
groq_api_key=st.sidebar.text_input("Enter your Groq api key:", type="password")



if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi I'm a chatbot who can search the web, How can I help you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:= st.chat_input(placeholder="What is Machine Learning ?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(api_key=groq_api_key,model="llama-3.1-8b-instant",streaming=True) # type: ignore
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response= search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)