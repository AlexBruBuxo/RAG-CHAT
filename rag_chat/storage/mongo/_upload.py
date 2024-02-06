import pandas as pd
import ast
from rag_chat.storage.mongo import mongodb_client

def str2list(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return None

def upload_data_to_mongodb(csv_file_path, database_name, collection_name, batch_size):
    df = pd.read_csv(csv_file_path)
    df['category'] = df['category'].apply(str2list)

    try:
        db = mongodb_client[database_name]
        collection = db[collection_name]

        for i in range(0, len(df), batch_size):
            try: 
                data_batch = df.iloc[i:i + batch_size].to_dict(orient='records')
                result = collection.insert_many(data_batch)
                print(f"Inserted batch {i // batch_size + 1}; uploaded {len(result.inserted_ids)} to collection {collection_name}")
            except Exception as e:
                print(e)
        print("Successfully uploaded all batches.")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    csv_file_path = 'data/products.csv'
    upload_data_to_mongodb(csv_file_path, 'products', 'data', batch_size=50)