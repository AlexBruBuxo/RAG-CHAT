import logging 
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter

from rag_chat.storage.mongo import mongodb_uri
from rag_chat.storage.storage import Storage
from rag_chat.storage.mongo.reader import CustomMongoReader
from rag_chat.storage.mongo.doc_store import load_doc_store
from rag_chat.storage.qdrant.vector_store import (
    load_vector_store,
    load_async_vector_store
)
from rag_chat.storage.config import (
    mongo_reader_config,
    sentence_splitter_config,
    metadata_extractors_config
)

logger = logging.getLogger(__name__)


def load_storage():
    node_parser = SentenceSplitter(**sentence_splitter_config)
    embedding = OpenAIEmbedding()
    vector_store = load_vector_store()
    doc_store = load_doc_store()

    storage = Storage(
        node_parser=node_parser, 
        embedding=embedding,
        vector_store=vector_store,
        doc_store=doc_store,
        metadata_extractors=metadata_extractors_config,
    )
    return storage

def load_async_storage():
    node_parser = SentenceSplitter(**sentence_splitter_config)
    embedding = OpenAIEmbedding()
    async_vector_store = load_async_vector_store()
    doc_store = load_doc_store()

    storage = Storage(
        node_parser=node_parser, 
        embedding=embedding,
        vector_store=async_vector_store,
        doc_store=doc_store,
        metadata_extractors=metadata_extractors_config,
    )
    return storage


if __name__ == "__main__":

    import time
    start_time = time.time()

    # Add documents
    reader = CustomMongoReader(uri=mongodb_uri)
    documents = reader.load_data(**mongo_reader_config)
    storage = load_storage()
    storage.add_docs(documents)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")

    # Add new document -> Works good (added on both the docstore and the vector_store)
    # reader = CustomMongoReader(uri=mongodb_uri)
    # mongo_reader_config['max_docs'] = 5
    # documents = reader.load_data(**mongo_reader_config)
    # documents = [documents[4]]
    # storage = load_storage()
    # storage.add_docs(documents)

    # Delete document (even the docs with multiple Nodes) -> Works good
    # storage = load_storage()
    # storage.delete_doc("d32c8b6f2ead69d851e336f5397ec527")

    # Upsert document (same document; not modified) -> Works good
    # reader = CustomMongoReader(uri=mongodb_uri)
    # mongo_reader_config['max_docs'] = 5
    # documents = reader.load_data(**mongo_reader_config)
    # documents = [documents[4]]
    # storage = load_storage()
    # storage.add_docs(documents)

    # Upsert document (modified metadata/content from document) -> Works good
    # reader = CustomMongoReader(uri=mongodb_uri)
    # mongo_reader_config['max_docs'] = 5
    # documents = reader.load_data(**mongo_reader_config)
    # documents[4].metadata["product_name"] = "Metadata has been modified!"
    # documents[4].text = documents[4].text + "The content has been modified!"
    # documents = [documents[4]]
    # storage = load_storage()
    # storage.add_docs(documents)