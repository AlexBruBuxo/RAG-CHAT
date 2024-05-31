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
