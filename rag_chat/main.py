import logging
from llama_index.llms import ChatMessage, MessageRole
from rag_chat.query.query import load_query_engine, load_retriever
from rag_chat.agent.chat import load_chat_engine
from rag_chat.query.query import load_async_query_engine
from rag_chat.log import setup_logging

# Note: This should be added at the start of the app to control the logging.
setup_logging()

# Note: This must be set at the start of each file when we want 
# to log something (this will show the name of the logger in the log)
logger = logging.getLogger(__name__)


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
    response = chat_engine.chat("Hi! Do you have any popcorn?") # get message
    # chat_engine.reset() # reset memory
    # chat_engine.stream_chat() # streaming
    print(response)
