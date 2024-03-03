from llama_index import get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from rag_chat.storage.load import load_storage, load_async_storage
from rag_chat.query.config import (
    similarity_top_k,
    node_postprocessors,
    response_sinthesizer
)

# If necessary more modularity, all this can be fully customized with a Query Pipeline
# https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/usage_pattern.html

def load_retriever():
    storage = load_storage()
    
    retriever = VectorIndexRetriever(
        index=storage.index,
        similarity_top_k=similarity_top_k,
    )
    return retriever

def load_query_engine():
    retriever = load_retriever()
    response_synthesizer = get_response_synthesizer(**response_sinthesizer)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors
    )
    return query_engine

def load_async_retriever():
    storage = load_async_storage()
    
    retriever = VectorIndexRetriever(
        index=storage.async_index,
        similarity_top_k=similarity_top_k,
    )
    return retriever

def load_async_query_engine():
    retriever = load_async_retriever()
    response_synthesizer = get_response_synthesizer(**response_sinthesizer)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        use_async=True,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors
    )
    return query_engine
