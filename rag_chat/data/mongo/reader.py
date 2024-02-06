import uuid
from typing import Dict, Iterable, List, Optional
from llama_index.readers import SimpleMongoReader
from llama_index.schema import Document, MetadataMode
from rag_chat.data.mongo import mongodb_uri

class CustomMongoReader(SimpleMongoReader):
    """Extends the SimpleMongoReader from llama-index.

    Concatenates each Mongo doc into Document used by LlamaIndex.

    Args:
        host (str): Mongo host.
        port (int): Mongo port.
        uri (str): Mongo uri.
    """

    def lazy_load_data(
        self,
        db_name: str,
        collection_name: str,
        field_names: List[str] = ["text"],
        separator: str = "",
        query_dict: Optional[Dict] = None,
        max_docs: int = 0,
        metadata_names: Optional[List[str]] = None,
        metadata_seperator: str = "\n",
        excluded_llm_metadata_keys: Optional[List[str]] = [],
        field_doc_id: Optional[str] = None
    ) -> Iterable[Document]:
        """Load data from the input directory.

        Args:
            db_name (str): name of the database.
            collection_name (str): name of the collection.
            field_names(List[str]): names of the fields to be concatenated.
                Defaults to ["text"]
            separator (str): separator to be used between fields.
                Defaults to ""
            query_dict (Optional[Dict]): query to filter documents. Read more
            at [official docs](https://www.mongodb.com/docs/manual/reference/method/db.collection.find/#std-label-method-find-query)
                Defaults to None
            max_docs (int): maximum number of documents to load.
                Defaults to 0 (no limit)
            metadata_names (Optional[List[str]]): names of the fields to be added
                to the metadata attribute of the Document. Defaults to None
            metadata_seperator (str): separator to be used between metadata fields.
                Defaults to "\n"
            excluded_llm_metadata_keys (Optional[List[str]]): names of the fields
                to be excluded when crafting the response. Defaults to []

        Returns:
            List[Document]: A list of documents.

        """
        db = self.client[db_name]
        cursor = db[collection_name].find(filter=query_dict or {}, limit=max_docs)

        for item in cursor:
            try:
                # Convert field to string to allow all types
                texts = [str(item[name]) for name in field_names]
            except KeyError as e:
                raise ValueError(
                    f"{e.args[0]} field not found in Mongo document."
                ) from e
            
            texts = self._flatten(texts)
            # Add field name to description
            text = separator.join(
                f"{field_name}: {text}" for field_name, text in zip(field_names, texts)
            )

            try:
                doc_id = item[field_doc_id]
            except KeyError as e:
                raise ValueError(
                    f"{e.args[0]} field not found in Mongo document."
                ) from e

            if metadata_names is None:
                yield Document(
                        id_=doc_id if doc_id is not None else str(uuid.uuid4()),
                        text=text, 
                        metadata_seperator=metadata_seperator,
                        excluded_llm_metadata_keys=excluded_llm_metadata_keys
                    )
            else:
                try:
                    metadata = {name: item[name] for name in metadata_names}
                except KeyError as err:
                    raise ValueError(
                        f"{err.args[0]} field not found in Mongo document."
                    ) from err
                
                # Transform 'category' to hierarchy if present
                if "category" in metadata:
                    metadata["category"] = " > ".join(metadata["category"])

                yield Document(
                        id_=doc_id if doc_id is not None else str(uuid.uuid4()),
                        text=text, 
                        metadata=metadata, 
                        metadata_seperator=metadata_seperator,
                        excluded_llm_metadata_keys=excluded_llm_metadata_keys
                    )



if __name__ == "__main__":

    import os
    from dotenv import load_dotenv

    load_dotenv()
    DB_NAME = os.getenv('MONGODB_DB_NAME')
    DATA_COLLECTION_NAME = os.getenv('MONGODB_DATA_COLLECTION_NAME')
    FIELD_NAMES = ["product_url", "product_name", "brand", "description", "available", 
                   "sale_price", "discount"]
    SEPARATOR = " \n\n"
    QUERY_DICT = {"description": { "$type": "string" }}
    MAX_DOCS = 50
    METADATA_NAMES = ["list_price", "category"]
    EXCLUDED_LLM_METADATA_KEYS = []
    FIELD_DOC_ID = "uniq_id"

    reader = CustomMongoReader(uri=mongodb_uri)
    documents = reader.load_data(
        DB_NAME, 
        DATA_COLLECTION_NAME, 
        FIELD_NAMES, 
        separator = SEPARATOR, 
        query_dict=QUERY_DICT,
        max_docs = MAX_DOCS,
        metadata_names = METADATA_NAMES,
        metadata_seperator = SEPARATOR,
        excluded_llm_metadata_keys = EXCLUDED_LLM_METADATA_KEYS,
        field_doc_id = FIELD_DOC_ID
    )

    print("The Document Object:\n")
    print(documents[:1])
    print("\n------------\n")

    print("What the embedding model will see when ranking the information:\n")
    print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))
    print("\n------------\n")

    print("What the LLM model will see when crafting the response:\n")
    print(documents[0].get_content(metadata_mode=MetadataMode.LLM))
