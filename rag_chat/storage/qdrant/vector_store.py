from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores import QdrantVectorStore
import os
from dotenv import load_dotenv
from rag_chat.storage.config import QDRANT_COLLECTION


load_dotenv()
qdrant_ip = os.getenv('QDRANT_IP')
qdrant_port = os.getenv('QDRANT_PORT')
qdrant_collection = QDRANT_COLLECTION

def load_vector_store():
    # client = QdrantClient(location=":memory:")
    client = QdrantClient(qdrant_ip, port=qdrant_port)
    vector_store = QdrantVectorStore(client=client, 
                                    collection_name=qdrant_collection)
    return vector_store

def load_async_vector_store():
    client = AsyncQdrantClient(qdrant_ip, port=qdrant_port)
    vector_store = QdrantVectorStore(aclient=client, 
                                    collection_name=qdrant_collection)
    return vector_store
