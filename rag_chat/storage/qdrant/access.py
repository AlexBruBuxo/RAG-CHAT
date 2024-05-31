from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import json
from dotenv import load_dotenv
from rag_chat.storage.config import QDRANT_COLLECTION


load_dotenv()
qdrant_ip = os.getenv('QDRANT_IP')
qdrant_port = os.getenv('QDRANT_PORT')
qdrant_collection = QDRANT_COLLECTION


client = QdrantClient(qdrant_ip, port=qdrant_port)


def get_node_by_string(string: str, collection_name: str):
    result = client.scroll(
        collection_name=collection_name,  #qdrant_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="_node_content",
                    match=models.MatchText(text=string),
                )
            ]
        )
    )
    return result

def get_product_name(string: str, collection_name: str, debug: bool = False):
    nodes = get_node_by_string(string, collection_name)
    if debug: [print(i+1, ": ", node, "\n\n") for i, node in enumerate(nodes)]

    product_names = []
    for node in nodes:
        if node:
            text = json.loads(node[0].payload["_node_content"])["text"]
            # print("TEXT: ", text)
            if "product_name: " in text:
                product_names.append(text.split('\n', 1)[0].replace("product_name: ", ""))
            else:
                print("No product name")
                product_names.append(text[:50])
    return product_names


def get_id_from_node(node):
    return json.loads(node[0].payload["_node_content"])["id_"]


if __name__ == "__main__":

    ## This script automates the process of translating Node IDs from one 
    ## retrieval evaluation dataset to another

    id = "8b5863d5-a3a3-414f-b87d-761baabc0487"
    source = '"id_": "' + id + '"'

    # Get product name from NodeID in the source collection
    product_names = get_product_name(source, "vector_store_2")
    if len(product_names) > 1:
        print("Multiple Nodes with the same ID")
    else:  
        # Get Node ID from the new collection
        # TODO: modify function to search on the metadata...
        print("Product name: ", product_names[0])
        nodes = get_node_by_string(product_names[0], "vector_store_4")
        for node in nodes:
            if node:
                id = get_id_from_node(node)
                print(id)
    

    # nodes = get_node_by_string("rail-guard securely", "vector_store")
    # print(nodes)