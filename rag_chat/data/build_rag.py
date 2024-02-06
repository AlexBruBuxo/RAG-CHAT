import logging 
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter

from rag_chat.data.mongo import mongodb_uri
from rag_chat.data.store import Store
from rag_chat.data.mongo.reader import CustomMongoReader
from rag_chat.data.mongo.doc_store import load_doc_store
from rag_chat.data.qdrant.vector_store import load_vector_store
from rag_chat.data.config import (
    mongo_reader_config,
    sentence_splitter_config,
    metadata_extractors_config
)

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    reader = CustomMongoReader(uri=mongodb_uri)
    try:
        documents = reader.load_data(**mongo_reader_config)
    except Exception as e:
        logger.error("Unable to read data from MongoDB:", e)

    node_parser = SentenceSplitter(**sentence_splitter_config)
    embedding = OpenAIEmbedding()
    vector_store = load_vector_store()
    doc_store = load_doc_store()

    store = Store(
        node_parser=node_parser, 
        embedding=embedding,
        vector_store=vector_store,
        doc_store=doc_store,
        metadata_extractors=metadata_extractors_config,
    )
