import os
from dotenv import load_dotenv

from llama_index.extractors import (
    SummaryExtractor,
    KeywordExtractor,
    EntityExtractor,
    TitleExtractor,
    QuestionsAnsweredExtractor,
)

load_dotenv()

DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the product:
{context_str}

Summarize the key features of the product. \
Do not mention the category, URL, price nor discount information. \

Summary: """


# MongoDB Data Reader
mongo_reader_config = {
    'db_name': os.getenv('MONGODB_DB_NAME'), 
    'collection_name': os.getenv('MONGODB_DATA_COLLECTION_NAME'), 
    'field_names': ["product_url", "product_name", "brand", "description", "available", "sale_price", "discount"], 
    'separator':  " \n\n", 
    'query_dict': {"description": { "$type": "string" }},
    'max_docs': 50,
    'metadata_names': ["list_price", "category"],
    'metadata_seperator': " \n\n",
    'excluded_llm_metadata_keys': [],
    'field_doc_id': "uniq_id"
}

# Sentence Splitter
sentence_splitter_config = {
    'chunk_size': 512,
    'chunk_overlap': 128
}

# Metadata Extractors
metadata_extractors_config = [
    # Only these two transformations require aprox 20-30s, which is 23h for 
    # 28000 entries...xD (and a lot of GPT calls)
    SummaryExtractor(
        summaries=["self"],
        prompt_template=DEFAULT_SUMMARY_EXTRACT_TEMPLATE
    ),
    KeywordExtractor(keywords=10),

    # TODO: Or create a custom extractor: https://docs.llamaindex.ai/en/latest/module_guides/indexing/metadata_extraction.html#custom-extractors 
    # Source code for extractors: https://github.com/run-llama/llama_index/blob/main/llama_index/extractors/metadata_extractors.py 
    # Can also be created easily using Pydantic: https://docs.llamaindex.ai/en/latest/examples/metadata_extraction/PydanticExtractor.html
    
    # TODO: QuestionAnsweredExtractor(): https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_metadata_extractor.html 
    # Example: https://docs.llamaindex.ai/en/latest/examples/metadata_extraction/MetadataExtractionSEC.html# 
    # QuestionsAnsweredExtractor(
    #     questions=3, llm=llm, metadata_mode=MetadataMode.EMBED
    # ),
]
