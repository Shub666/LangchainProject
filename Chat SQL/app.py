import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq


st.set_page_config(page_title="Langchain: Chat with SQL")
st.title("Langchain: Chat with SQl DB")

INJECTION_WARNING="""
                SQL agent can be vulnerable to prompt injection. Use a DB role
"""

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

radio_opt = ["Use SQLLite 3 DB - Student.db","Connect to your SQL DB"]

selected_opt=st.sidebar.radio(label="Choose the Db",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mqsql_host=st.sidebar.text_input("My sql host name")
    mqsql_user=st.sidebar.text_input("My sql user name")
    mqsql_pwd=st.sidebar.text_input("My sql password",type="password")
    mysql_db=st.sidebar.text_input("My sql Database")
else:
    db_uri=LOCALDB

gorq_api_key=st.sidebar.text_input(label="GROQ API key", type="password")

if not db_uri:
    st.info("Please enter DB info")
if not gorq_api_key:
    st.info("Please enter GROQ key")

llm=ChatGroq(api_key=gorq_api_key,model="llama-3.3-70b-versatile",streaming=True) # type: ignore

@st.cache_resource(ttl="2h")
def confgiure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"student.db").absolute()
        print(dbfilepath)
        creator=lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri==MYSQL:
        if not (mysql_db and mysql_host and mysql_password and mysql_user):
            st.error("Please provide all MySQL connetion deatils.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_uri==MYSQL:
    db=confgiure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_dd) # type: ignore
else:
    db=confgiure_db(db_uri)

toolkit= SQLDatabaseToolkit(db=db,llm=llm) # type: ignore

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state["messages"] = [{"role":"assistant","content":"How Can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

user_query= st.chat_input(placeholder="Ask any thing from DB")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback=StreamlitCallbackHandler(st.container())
        response = agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
