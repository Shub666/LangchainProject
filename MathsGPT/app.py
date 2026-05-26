import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains.llm_math.base import LLMMathChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool,initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text to Match Solver and Data search assistant")
st.title("Text to Match Solver and Data search assistant")

g_api_key=st.sidebar.text_input(label="Groq API Key", value="", type="password")

if not g_api_key:
    st.info("Please add your Groq Key")
    st.stop()

llm= ChatGroq(api_key=g_api_key,model="llama-3.3-70b-versatile") # type: ignore


## Tools

wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet"
)

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related problems"
)

prompt = """
You are agent task for solving user math questions. Logically arriave at the solution and provied details explanation and display point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool= Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answer logic based and reasoing questions."
)


assitant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm your Math chatbot"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# def generate_response(question):
#     response=assitant_agent.invoke({'input':question})
#     return response

question=st.text_area("Enter your question")
if st.button("Find my Ans"):
    if question:
        with st.spinner("Generate Response:"):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb= StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assitant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assitant","content":response})
            st.write('### Response:')
            st.success(response)
    else:
        st.warning("Please input the question")