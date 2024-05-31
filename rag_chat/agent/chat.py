from typing import List
from llama_index.llms import ChatMessage


def load_chat_engine(
        chat_mode: str="react", 
        chat_history: List[ChatMessage] = [],
        verbose: bool = False,
        **kwargs
    ):
    retriever = kwargs.get('retriever', None)
    query_engine = kwargs.get('query_engine', None)

    if chat_mode == "react":
        if query_engine:
            from rag_chat.agent.react import ReActChatEngine
            react_chat_engine = ReActChatEngine(
                chat_history=chat_history,
                query_engine=query_engine,
                verbose=verbose
            )
            return react_chat_engine.load_engine()
        else:
            raise Exception(f"'query_engine' is a required parameter for 'react' chat mode.")
    
    elif chat_mode == "condense":
        if query_engine:
            from rag_chat.agent.condense import CondenseChatEngine
            condense_chat_engine = CondenseChatEngine(
                chat_history=chat_history,
                query_engine=query_engine,
                verbose=verbose
            )
            return condense_chat_engine.load_engine()
        else:
            raise Exception(f"'query_engine' is a required parameter for 'condense' chat mode.")
    
    elif chat_mode == "condense_context":
        if query_engine and retriever:
            from rag_chat.agent.condense_context import CondenseContextChatEngine
            condense_context_chat_engine = CondenseContextChatEngine(
                chat_history=chat_history,
                query_engine=query_engine,
                retriever=retriever,
                verbose=verbose
            )
            return condense_context_chat_engine.load_engine()
        else:
            raise Exception(f"'query_engine' and 'retriever' are a required parameter for 'condense_context' chat mode.")
    
    else:
        raise Exception(f"{str(chat_mode)} is not a valid mode.")
