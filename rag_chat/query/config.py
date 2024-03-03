from typing import List
from pydantic import BaseModel
from llama_index.response_synthesizers import ResponseMode
from llama_index.postprocessor import (
    SimilarityPostprocessor, 
    LongContextReorder
)

# Vector Index Retriver
similarity_top_k=5

# Node Postprocessors
node_postprocessors = [
    LongContextReorder(),
    SimilarityPostprocessor(similarity_cutoff=0.7)
    # TODO: Since we have PREVIOUS/NEXT fields in metadata, we can add 
    # postprocessors that take this into account (but should only apply to 
    # Nodes comming form the same doc_id) -> For example, if the top Nodes have PREVIOUS/NEXT with the same ID, then fetch them also.
    # https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/node_postprocessors.html# 
]

# Pydantic response model
class Response(BaseModel):
    """Data model for a response."""
    product_urls: List[str]
    answer: str

# Response synthesizer
response_sinthesizer = {
    'response_mode': ResponseMode.COMPACT,
    # verbose: True,

    # TODO: We can consider a custom synthesizer:
    # https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#custom-response-synthesizers

    # TODO: Try pydantic extractor
    # output_cls=Response

    # TODO: toggle with this: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#using-structured-answer-filtering 
    # structured_answer_filtering=True
}