from llama_index.llms import ChatMessage, MessageRole
from rag_chat.query.query import load_query_engine, load_retriever
from rag_chat.agent.chat import load_chat_engine

from rag_chat.query.query import load_async_query_engine


SYSTEM_MESSAGE = """\
You are a helpful assistant for an online retail shop.\
Your job is to help customers with product queries.\
Be friendly, helpful and only provide information related to the shopping process.
"""


retriever = load_retriever()
query_engine = load_query_engine()


chat_history = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=SYSTEM_MESSAGE
    )
]

chat_engine = load_chat_engine(
    chat_mode="react",
    chat_history=chat_history,
    retriever=retriever,
    query_engine=query_engine,
    verbose=True
)


if __name__ == "__main__":
    chat_engine.chat_repl()
