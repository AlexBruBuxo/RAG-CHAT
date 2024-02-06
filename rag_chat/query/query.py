from typing import List
from pydantic import BaseModel
from llama_index import get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import ResponseMode
from llama_index.postprocessor import (
    SimilarityPostprocessor, 
    LongContextReorder
)

from rag_chat.storage.load import load_storage


storage = load_storage()

retriever = VectorIndexRetriever(
    index=storage.index,
    similarity_top_k=5,
)

node_postprocessors = [
    LongContextReorder(),
    SimilarityPostprocessor(similarity_cutoff=0.7)
    # TODO: if we have PREVIOUS/NEXT fields in metadata, we can add 
    # postprocessors that take this into account (but should only apply to 
    # Nodes comming form the same doc_id)
    # https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/node_postprocessors.html# 
]

class Response(BaseModel):
    """Data model for a response."""
    product_urls: List[str]
    answer: str

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    # verbose=True,

    # TODO: We can consider a custom synthesizer:
    # https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#custom-response-synthesizers

    # TODO: Try pydantic extractor
    # output_cls=Response

    # TODO: toggle with this: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#using-structured-answer-filtering 
    # structured_answer_filtering=True
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=node_postprocessors
)
