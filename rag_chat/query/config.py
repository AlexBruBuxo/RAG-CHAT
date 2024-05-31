from typing import List
from pydantic import BaseModel
from llama_index.response_synthesizers import ResponseMode
from llama_index.postprocessor import (
    SimilarityPostprocessor, 
    LongContextReorder
)
from rag_chat.utils import read_inference_conf


inference_conf = read_inference_conf()

# Vector Index Retriver
similarity_top_k = inference_conf["similarity_top_k"]

# Node Postprocessors
node_postprocessors = []
if inference_conf["include_long_context_reorder"]:
    node_postprocessors.append(
        LongContextReorder()
    )
if inference_conf["include_similarity_postprocessor"]:
    node_postprocessors.append(
        SimilarityPostprocessor(similarity_cutoff=inference_conf["similarity_cutoff"])
    )
# NOTE: Since we have PREVIOUS/NEXT fields in metadata, we can add 
# postprocessors that take this into account (but should only apply to 
# Nodes comming form the same doc_id)
# For example, if the top Nodes have PREVIOUS/NEXT with the same ID, then 
# fetch them also: https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/node_postprocessors.html# 

# Pydantic response model
class Response(BaseModel):
    """Data model for a response."""
    product_urls: List[str]
    answer: str

# Response synthesizer
response_sinthesizer = {
    'response_mode': ResponseMode.COMPACT, # ResponseMode.REFINE, 
    'verbose': inference_conf["verbose"],
    # NOTE: We can consider a custom synthesizer:
    # https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#custom-response-synthesizers
}
if inference_conf["pydantic_extractor"]:
    response_sinthesizer['output_cls'] = Response  # Pydantic extractor
if inference_conf["structured_answer_filtering"]:
    response_sinthesizer["structured_answer_filtering"] = True  # https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#using-structured-answer-filtering