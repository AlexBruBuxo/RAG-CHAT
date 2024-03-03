from typing import List
from llama_index.llms import ChatMessage
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.prompts import PromptTemplate
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from rag_chat.agent.config import (
    CONDENSE_PROMPT,
)



class CondenseChatEngine():
    def __init__(self,
            chat_history: List[ChatMessage],
            query_engine: RetrieverQueryEngine,
            verbose: bool = False
        ):
        self.chat_history = chat_history
        self.query_engine = query_engine
        self.verbose = verbose
    
    def load_engine(self):
        """Load Condense chat engine."""
        return CondenseQuestionChatEngine.from_defaults(
            query_engine=self.query_engine,
            condense_question_prompt=PromptTemplate(CONDENSE_PROMPT),
            chat_history=self.chat_history,
            verbose=self.verbose
        )
