from pymongo.mongo_client import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGODB_USER = os.getenv('MONGODB_USER')
MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD')
MONGODB_CLUSTER = os.getenv('MONGODB_CLUSTER')

mongodb_uri = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_CLUSTER}/?retryWrites=true&w=majority"
mongodb_client = MongoClient(mongodb_uri, connectTimeoutMS=30000)


if __name__ == "__main__":
    # Test connection
    try:
        mongodb_client.admin.command('ping')
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(e)