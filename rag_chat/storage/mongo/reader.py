import uuid
from typing import Dict, Iterable, List, Optional
from llama_index.readers import SimpleMongoReader
from llama_index.schema import Document, MetadataMode

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
        skip_docs: int = 0,
        max_docs: int = 0,
        metadata_names: Optional[List[str]] = None,
        metadata_seperator: str = "\n",
        excluded_llm_metadata_keys: Optional[List[str]] = [],
        excluded_embed_metadata_keys: Optional[List[str]] = [],
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
            skip_docs (int): number of documents to skip.
                Defaults to 0
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
        # cursor = db[collection_name].find(filter=query_dict or {}, limit=max_docs)
        cursor = db[collection_name].find(filter=query_dict or {}).skip(skip_docs).limit(max_docs)

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
                        excluded_llm_metadata_keys=excluded_llm_metadata_keys,
                        excluded_embed_metadata_keys=excluded_embed_metadata_keys
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

    from rag_chat.storage.mongo import mongodb_uri
    from rag_chat.storage.config import mongo_reader_config

    mongo_reader_config['max_docs'] = 2

    reader = CustomMongoReader(uri=mongodb_uri)
    documents = reader.load_data(**mongo_reader_config)

    print("The Document Object:\n")
    [print(doc.id_) for doc in documents]

    # In our case, metadata is always added to the content.
    print("\n--------------------------------------------------\n")
    print("Ranking: what the embedding model will see\n")
    print(documents[0].get_content(metadata_mode=MetadataMode.EMBED))
    print("\n--------------------------------------------------\n")
    print("Crafting response: what the LLM model will see:\n")
    print(documents[0].get_content(metadata_mode=MetadataMode.LLM))
    print("\n--------------------------------------------------\n")
    print(documents[0].get_content(metadata_mode=MetadataMode.ALL))

    print("\n--------------------------------------------------\n")
    print(documents[0].get_content(metadata_mode=MetadataMode.NONE))
