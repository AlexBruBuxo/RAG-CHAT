import json
from typing import List
from llama_index import Document
from llama_index.llama_dataset.generator import RagDatasetGenerator
from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.llms.openai import OpenAI
from llama_index import ServiceContext

from rag_chat.storage.mongo.reader import CustomMongoReader
from rag_chat.storage.mongo.client import mongodb_uri
from rag_chat.evaluation.config import (
    eval_mongo_reader_config,
    NUM_QUESTIONS_PER_NODE,
    DATASET_FILE_PATH,
    GENERATE_DATASET,
    SAVE_DATASET,
    SHOW_PROGRESS
)


DEFAULT_NUM_QUESTIONS_PER_NODE = 2

QUESTION_GENERATION_PROMPT = """\
Product information/context is below.
---------------------
{context_str}
---------------------
Given the product information and not prior knowledge generate only questions.
The questions should be diverse in nature. A question can be specific to the \
product, such as: What is the price for the Kajambo Picture Frames Set?. The \
question can also be generic to the store, such as: Do you have any shoes \
for sport?.

Generate questions based on the below query.
{query_str}
"""

QUESTION_GEN_QUERY = f"""\
You are a customer of an online store. Your task is to setup \
{NUM_QUESTIONS_PER_NODE} questions for the store's chatbot. Restrict the \
questions to the context information provided.
"""

TEXT_QA_PROMPT = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, 
answer the query.
Query: {query_str}
Answer: 
"""


class EvalDataset():
    """RAG Evaluation dataset (question-answer pairs).

    Args:
        queries (List[str]): List of queries. 
        answers (List[str]): List of refrence answers. 
        contexts (List[str]): List of reference contexts. 

    """

    def __init__(
        self, 
        queries: List[str] = [], 
        answers: List[str] = [],
        contexts: List[str] = []
    ) -> None:
        self.queries = queries
        self.answers = answers
        self.contexts = contexts
    
    @classmethod
    def generate(
        cls,
        documents: List[Document],
        service_context: ServiceContext,
        num_questions_per_node: int = DEFAULT_NUM_QUESTIONS_PER_NODE,
        show_progress: bool = True,
    ) -> 'EvalDataset':
        """generate dataset from documents."""
        dataset_generator = RagDatasetGenerator.from_documents(
            documents,
            service_context=service_context,
            num_questions_per_chunk=num_questions_per_node,
            show_progress=show_progress,
            text_question_template=PromptTemplate(QUESTION_GENERATION_PROMPT),
            text_qa_template=PromptTemplate(
                TEXT_QA_PROMPT, prompt_type=PromptType.QUESTION_ANSWER
            ),
            question_gen_query=QUESTION_GEN_QUERY
        )
        rag_dataset = dataset_generator.generate_dataset_from_nodes()
        return cls(
            queries = [e.query for e in rag_dataset.examples],
            answers = [e.reference_answer for e in rag_dataset.examples],
            contexts = [e.reference_contexts for e in rag_dataset.examples]
        )   
    
    def save_json(
            self, 
            file_path: str
        ) -> None:
        """Save dataset to JSON file."""
        data = {
            'queries': self.queries,
            'answers': self.answers,
            'contexts': self.contexts
        }
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)

    @classmethod
    def from_json(
        cls,
        file_path: str
    ) -> 'EvalDataset':
        """Load dataset from JSON file."""
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return cls(
            queries=data['queries'],
            answers=data['answers'],
            contexts=data['contexts']
        )


if __name__ == "__main__":

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo") # Set gpt-3.5 due to rate limit being exceeded
    reader = CustomMongoReader(uri=mongodb_uri)
    documents = reader.load_data(**eval_mongo_reader_config)
    eval_service_context = ServiceContext.from_defaults(llm=llm)

    # Generate and save dataset
    if GENERATE_DATASET and len(documents) > 0:
        print("Generating evaluation dataset...") # TODO: change to log.info
        eval_dataset = EvalDataset.generate(
            documents=documents,
            service_context=eval_service_context,
            num_questions_per_node=NUM_QUESTIONS_PER_NODE,
            show_progress=SHOW_PROGRESS
        )
        if SAVE_DATASET and DATASET_FILE_PATH != "":
            print("Saving evaluation dataset...") # TODO: change to log.info
            eval_dataset.save_json(
                file_path=DATASET_FILE_PATH
            )