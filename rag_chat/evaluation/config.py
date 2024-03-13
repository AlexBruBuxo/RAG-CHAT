from rag_chat.storage.config import mongo_reader_config
from rag_chat.utils import read_inference_conf

inference_conf = read_inference_conf()

# Eval Dataset
MAX_DOCS = inference_conf["max_docs"]  # Number of documents to build eval dataset
NUM_QUESTIONS_PER_NODE = inference_conf["num_questions_per_node"]
DATASET_FILE_PATH = inference_conf["dataset_file_path"]
RETRIEVAL_DATASET_FILE_PATH = inference_conf["retrieval_dataset_file_path"]
GENERATE_DATASET = inference_conf["generate_dataset"]
SAVE_DATASET = inference_conf["save_dataset"]
SHOW_PROGRESS = inference_conf["show_progress"]

# Evaluation
CHAT_MODE = inference_conf["chat_mode"]
SAVE_EVAL_DATAFRAME = inference_conf["save_eval_dataframe"]
SAVE_SUMMARY_STATISTICS_DATAFRAME = inference_conf["save_summary_statistics_dataframe"]
base_folder = inference_conf["experiment"]
EVAL_DATAFRAME_FILE_PATH = base_folder + inference_conf["eval_dataframe_file_path"]
RETRIEVAL_EVAL_DATAFRAME_FILE_PATH = base_folder + inference_conf["retrieval_eval_dataframe_file_path"]
SUMMARY_STATISTICS_DATAFRAME_FILE_PATH = base_folder + inference_conf["summary_statistics_dataframe_file_path"]
INDEX_CONFIGURATION_FOLDER_PATH = inference_conf["experiment"]
INFERENCE_CONFIGURATION_FOLDER_PATH = inference_conf["experiment"]


eval_mongo_reader_config = mongo_reader_config.copy()
eval_mongo_reader_config["max_docs"] = MAX_DOCS
