from llama_index.storage.docstore import MongoDocumentStore
from rag_chat.storage.mongo import mongodb_uri
import os
from dotenv import load_dotenv
from rag_chat.storage.config import MONGODB_DOCSTORE_COLLECTION_NAME


load_dotenv()
db_name = os.getenv('MONGODB_DB_NAME')
docstore_collection_name = MONGODB_DOCSTORE_COLLECTION_NAME

def load_doc_store():
    doc_store = MongoDocumentStore.from_uri(
            uri=mongodb_uri,
            db_name=db_name,
            namespace=docstore_collection_name
        )
    return doc_store
