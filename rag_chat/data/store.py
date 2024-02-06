import logging
from typing import Dict, List, Optional
from pydantic.v1.main import ModelMetaclass

from llama_index import VectorStoreIndex
from llama_index.schema import MetadataMode, Document, TextNode
from llama_index.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.vector_stores.types import BasePydanticVectorStore
from llama_index.storage.kvstore.mongodb_kvstore import MongoDBKVStore


logger = logging.getLogger(__name__)


class Store:
    def __init__(
        self,
        node_parser: ModelMetaclass,
        embedding: ModelMetaclass,
        vector_store: BasePydanticVectorStore,
        doc_store: MongoDBKVStore,
        metadata_extractors: Optional[List[ModelMetaclass]] = [],
    ) -> None:
        """Initialize with parameters."""
        self.node_parser = node_parser
        self.embedding = embedding
        self.metadata_extractors = metadata_extractors
        self._vector_store = vector_store
        self._doc_store = doc_store
    
    @property
    def pipeline(self) -> IngestionPipeline:
        transformations = [
            self.node_parser,
            *self.metadata_extractors,
            self.embedding,
        ]
        if self._vector_store is not None:
            return IngestionPipeline(
                transformations=transformations,
                vector_store=self._vector_store,
                docstore=self._doc_store,
                docstore_strategy=DocstoreStrategy.UPSERTS,
            )
        else:
            raise ValueError("Vector store cannot be None.")

    @property
    def index(self) -> VectorStoreIndex:
        try:
            return VectorStoreIndex.from_vector_store(
                self.pipeline.vector_store,
                # use_async=True -> Async add nodes to index??
            ) 
        except Exception as e:
            logger.error("Unable to load vector store:", e)

    def add_docs(self, documents: List[Document]) -> List[TextNode]:
        """Add documents to index; create index if it does not exist."""
        # TODO: This fails to handle deduplicates on the Vvector_store, only the docstore
        # Should not happen: https://docs.llamaindex.ai/en/stable/examples/ingestion/redis_ingestion_pipeline.html
        try:
            return self.pipeline.run(documents=documents)
        except Exception as e:
            logger.error("Unable to create index:", e)
    
    def delete_doc(self, doc_id):
        try:
            self.index.delete_ref_doc(doc_id, delete_from_docstore=False)
            self.pipeline.docstore.delete_document(doc_id)
        except Exception as e:
            logger.error("Unable to delete document from index:", e)

    def clear_vector_store(self):
        """Clear vector store."""
        if self.pipeline.vector_store is not None and self.pipeline.vector_store._index_exists():
            self.pipeline.vector_store.delete_index()


# TODO: Improve Index: https://docs.llamaindex.ai/en/latest/module_guides/indexing/index_guide.html#
# 1) Check VectorStoreIndex: https://docs.llamaindex.ai/en/latest/module_guides/indexing/vector_store_index.html
# 2) Check Keyword Table Index: https://docs.llamaindex.ai/en/latest/api_reference/indices/table.html# 
# 3) Custom: combine both: https://docs.llamaindex.ai/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html# 
# 4) Check also Knowledge Graphs: https://docs.llamaindex.ai/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphDemo.html# 
