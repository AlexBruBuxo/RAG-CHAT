import os
from dotenv import load_dotenv
from rag_chat.utils import read_index_conf

from llama_index.extractors import (
    SummaryExtractor,
    KeywordExtractor,
    EntityExtractor,
    TitleExtractor,
    QuestionsAnsweredExtractor,
)



load_dotenv()
index_conf = read_index_conf()

DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
Here is the information of a product:
{context_str}

Summarize the key features of the product. \
Do not mention the category, URL, price nor discount information. \

Summary: """



# MongoDB Data Reader
mongo_reader_config = {
    'db_name': os.getenv('MONGODB_DB_NAME'), 
    'collection_name': os.getenv('MONGODB_DATA_COLLECTION_NAME'), 
    'field_names': ["product_name", "brand", "category", "product_url", "description", "available", "sale_price", "list_price", "discount"], 
    'separator':  " \n\n", 
    'query_dict': {"description": { "$type": "string" }},
    'max_docs': index_conf["max_docs"],
    'metadata_names': [],
    'metadata_seperator': " \n\n",
    'excluded_llm_metadata_keys': [], # What the LLM won't see when crafting the response
    'excluded_embed_metadata_keys': [], # What the Emmbeding model won't see when ranking Nodes
    'field_doc_id': "uniq_id"
}


# Mongo Data Writer
MONGODB_DOCSTORE_COLLECTION_NAME = index_conf["mongodb_docstore_collection_name"]



# Sentence Splitter
sentence_splitter_config = {
    'chunk_size': index_conf["chunk_size"],
    'chunk_overlap': index_conf["chunk_overlap"],
    'include_prev_next_rel': index_conf["inlcude_prev_next_rel"]  # Default True 
}



# Metadata Extractors
metadata_extractors_config = []
# Only these two transformations require aprox 20-30s for 200 elements, 
# which is 23h for 28000 entries... (and a lot of GPT calls)
if index_conf["include_summary_extractor"]:
    metadata_extractors_config.append(
        SummaryExtractor(
            summaries=[index_conf["summaries"]],
            prompt_template=DEFAULT_SUMMARY_EXTRACT_TEMPLATE
        )
    )
if index_conf["include_keyword_extractor"]:
    metadata_extractors_config.append(
        KeywordExtractor(keywords=index_conf["keywords"])
    )

# NOTE: Or create a custom extractor: https://docs.llamaindex.ai/en/latest/module_guides/indexing/metadata_extraction.html#custom-extractors 
# Source code for extractors: https://github.com/run-llama/llama_index/blob/main/llama_index/extractors/metadata_extractors.py 
# Can also be created easily using Pydantic: https://docs.llamaindex.ai/en/latest/examples/metadata_extraction/PydanticExtractor.html
    
# NOTE: QuestionAnsweredExtractor(): https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_metadata_extractor.html 
# Example: https://docs.llamaindex.ai/en/latest/examples/metadata_extraction/MetadataExtractionSEC.html# 
# QuestionsAnsweredExtractor(
#     questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
# ),



# Qdrant
QDRANT_COLLECTION = index_conf["qdrant_collection"]