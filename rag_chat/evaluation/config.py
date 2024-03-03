from rag_chat.storage.config import mongo_reader_config


MAX_DOCS = 20  # Number of documents to use to build the eval dataset
NUM_QUESTIONS_PER_NODE = 3
DATASET_FILE_PATH = "data/eval_dataset.json"
GENERATE_DATASET = True
SAVE_DATASET = True
SHOW_PROGRESS = True
CHAT_MODE = "react"
SAVE_EVAL_DATAFRAME = True
SAVE_SUMMARY_STATISTICS_DATAFRAME = True
EVAL_DATAFRAME_FILE_PATH = "experiments/v1/eval_results.csv"
SUMMARY_STATISTICS_DATAFRAME_FILE_PATH = "experiments/v1/summary_statistics.csv"

eval_mongo_reader_config = mongo_reader_config.copy()
eval_mongo_reader_config["max_docs"] = MAX_DOCS
