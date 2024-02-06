from llama_index.storage.docstore import MongoDocumentStore
from rag_chat.data.mongo import mongodb_uri
import os
from dotenv import load_dotenv

load_dotenv()
db_name = os.getenv('MONGODB_DB_NAME')
docstore_collection_name = os.getenv('MONGODB_DOCSTORE_COLLECTION_NAME')

def load_doc_store():
    doc_store = MongoDocumentStore.from_uri(
            uri=mongodb_uri,
            db_name=db_name,
            namespace=docstore_collection_name
        )
    return doc_store
