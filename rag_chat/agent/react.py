from typing import List
from llama_index.llms import ChatMessage
from llama_index.agent import ReActAgent
from llama_index.tools import QueryEngineTool
from llama_index.memory.types import BaseMemory
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.agent.react.formatter import ReActChatFormatter

from rag_chat.agent.config import (
    REACT_CHAT_SYSTEM_HEADER,
    CONTEXT_REACT_CHAT_SYSTEM_HEADER,
    QUERY_ENGINE_TOOL_NAME,
    QUERY_ENGINE_TOOL_DESCRIPTION,
)



class ReActChatEngine():
    def __init__(self,
            chat_history: List[ChatMessage],
            query_engine: RetrieverQueryEngine,
            verbose: bool = False
        ):
        self.chat_history = chat_history
        self.query_engine = query_engine
        self.verbose = verbose

    @property
    def query_engine_tool(self):
        """Convert Query Engine to Query Engine Tool."""
        return QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name=QUERY_ENGINE_TOOL_NAME,
            description=QUERY_ENGINE_TOOL_DESCRIPTION
        )
    
    @property
    def chat_formatter(self):
        """ReAct chat formatter."""
        return ReActChatFormatter.from_defaults(
            system_header=REACT_CHAT_SYSTEM_HEADER,
            context=CONTEXT_REACT_CHAT_SYSTEM_HEADER
        )

    @property
    def memory(self):
        return BaseMemory.from_defaults(chat_history=self.chat_history)

    def load_engine(self):
        """Load ReAct chat agent."""
        return ReActAgent.from_tools(
            tools=[self.query_engine_tool],
            memory=self.memory,
            max_iterations=10, # default 10
            verbose=self.verbose,
            react_chat_formatter=self.chat_formatter,
        )


# This would use the OpenAIAgent by default
# AgentRunner.from_llm(
#     tools=[query_engine_tool],
# )
# OpenAIAgent.from_llm(
#     tools=[query_engine_tool],
# )