from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import OpenAI
from llama_index.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaIndexTool,
)

from rag_chat.query.query import load_query_engine



query_engine = load_query_engine()

# ReAct with LangChain
tool_config = IndexToolConfig(
    query_engine=query_engine,
    name=f"Vector Index",
    description=f"useful for when you want to answer queries about the products in the store.",
    tool_kwargs={"return_direct": True},
)

tools = [LlamaIndexTool.from_tool_config(tool_config)]

# NOTE: we can modify ReAct prompt
prompt = hub.pull("hwchase17/react-chat")
# print(prompt)

llm = OpenAI()

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke(
    {
        "input": "Do you have any popcorn?",
        "chat_history": ""
    }
))