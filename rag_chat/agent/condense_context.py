from typing import List
from llama_index.llms import ChatMessage
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.chat_engine.condense_plus_context import CondensePlusContextChatEngine

from rag_chat.agent.config import (
    CONDENSE_PROMPT,
    CONTEXT_PROMPT
)



class CondenseContextChatEngine():
    def __init__(self,
            chat_history: List[ChatMessage],
            query_engine: RetrieverQueryEngine,
            retriever: VectorIndexRetriever,
            verbose: bool = False
        ):
        self.chat_history = chat_history
        self.query_engine = query_engine
        self.retriever = retriever
        self.verbose = verbose
    
    def load_engine(self):
        """Load Condense Plus Context chat engine."""
        return CondensePlusContextChatEngine.from_defaults(
            query_engine=self.query_engine,
            retriever=self.retriever,
            context_prompt=CONTEXT_PROMPT,
            condense_prompt=CONDENSE_PROMPT,
            chat_history=self.chat_history,
            verbose=self.verbose
        )
