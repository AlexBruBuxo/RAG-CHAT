import warnings
import time
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from llama_index import Document
from llama_index.llms.openai import OpenAI
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.chat_engine.types import AgentChatResponse, BaseChatEngine
from llama_index.embeddings import SimilarityMode
from llama_index.evaluation import (
    RelevancyEvaluator,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    SemanticSimilarityEvaluator,
)

from rag_chat.storage.mongo.reader import CustomMongoReader
from rag_chat.storage.mongo.client import mongodb_uri
from rag_chat.evaluation.dataset import EvalDataset
from rag_chat.query.query import load_async_query_engine, load_retriever
from rag_chat.agent.chat import load_chat_engine
from rag_chat.evaluation.config import (
    eval_mongo_reader_config,
    NUM_QUESTIONS_PER_NODE,
    DATASET_FILE_PATH,
    GENERATE_DATASET,
    SAVE_DATASET,
    SHOW_PROGRESS,
    CHAT_MODE,
    SAVE_EVAL_DATAFRAME,
    EVAL_DATAFRAME_FILE_PATH,
    SAVE_SUMMARY_STATISTICS_DATAFRAME,
    SUMMARY_STATISTICS_DATAFRAME_FILE_PATH
)


DEFAULT_NUM_QUESTIONS_PER_NODE = 2
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD = 0.8



def custom_parser(eval_response: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Default parser function for evaluation response.

    Args:
        eval_response (str): The response string from the evaluation.

    Returns:
        Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.
    """
    score_str, reasoning_str = eval_response.strip().split("\n", 1)
    try:
        score = float(score_str)
    except Exception as e:
        print("Unable to get score.")
        # TODO: need to check how to handle this
        score = 0
    reasoning = reasoning_str.lstrip("\n")
    return score, reasoning


class Eval():
    def __init__(
            self,
            results: List[Dict] = [], 
            elapsed_times: float = None,
            llm: OpenAI = OpenAI(temperature=0, model="gpt-4")
        ):
        self.results = results
        self.elapsed_times = elapsed_times
        self.llm = llm
    
    @staticmethod
    def get_eval_dataset(
        dataset_file_path: str = "",
    ) -> EvalDataset:
        """Load evaluation from json file."""
        print("Loading evaluation dataset...") # TODO: change to log.info
        eval_dataset = EvalDataset.from_json(
            file_path=dataset_file_path
        )
        return eval_dataset

    def get_df(
            self,
            save_csv: bool = False, 
            csv_file_path: str = None
        ) -> pd.DataFrame:
        
        data = []
        for time, result in zip(self.elapsed_times, self.results):
            row = {
                    "Time": time,
                    "Query": result["relevancy"].query,
                    "Response": result["relevancy"].response,
                    "Relevancy_Passing": result["relevancy"].passing,
                    "Relevancy_Feedback": result["relevancy"].feedback,
                    "Relevancy_Invalid": result["relevancy"].invalid_result,
                    "Relevancy_Invalid_Reason": result["relevancy"].invalid_reason,
                    "Faithfulness_Passing": result["faithfulness"].passing,
                    "Faithfulness_Feedback": result["faithfulness"].feedback,
                    "Faithfulness_Score": result["faithfulness"].score,
                    "Faithfulness_Invalid": result["faithfulness"].invalid_result,
                    "Faithfulness_Invalid_Reason": result["faithfulness"].invalid_reason,
                    "Correctness_Passing": result["correctness"].passing,
                    "Correctness_Feedback": result["correctness"].feedback,
                    "Correctness_Score": result["correctness"].score,
                    "Correctness_Feedback": result["correctness"].feedback,
                    "Correctness_Invalid": result["correctness"].invalid_result,
                    "Correctness_Invalid_Reason": result["correctness"].invalid_reason,
                    "Semantic_Similarity_Passing": result["semantic_similarity"].passing,
                    "Semantic_Similarity_Score": result["semantic_similarity"].score,
                    "Response_Source": result["relevancy"].contexts,
                }
            data.append(row)
        df = pd.DataFrame(data)

        if save_csv:
            if csv_file_path:
                df.to_csv(csv_file_path, index=False)
            else:
                print("Unable to save Evaluation DataFrame to CSV.")

        return df
    
    def get_summary_statistics(
            self,
            save_csv: bool = False, 
            csv_file_path: str = None
        ) -> pd.DataFrame:
        relevant_metrics = {
            "faithfulness": "float",
            "correctness": "float",
            "semantic_similarity": "float",
            "relevancy": "bool",
            "faithfulness": "bool",
            "correctness": "bool",
            "semantic_similarity": "bool"
        }

        summary_stats = {}
        for metric, dtype in relevant_metrics.items():
            values = []
            for result in self.results:
                if hasattr(result[metric], 'score'):
                    values.append(result[metric].score)
                elif hasattr(result[metric], 'passing'):
                    # Convert "TRUE" and "FALSE" strings to boolean values
                    values.append(result[metric].passing.upper() == "TRUE")

            if dtype == "float":
                summary_stats[metric + "_score_mean"] = np.mean(values)
                summary_stats[metric + "_score_variance"] = np.var(values)
                summary_stats[metric + "_score_p90"] = np.percentile(values, 90)
            elif dtype == "bool":
                summary_stats[metric + "_passing_rate"] = sum(values) / len(values) * 100 if values else 0

        summary_df = pd.DataFrame(summary_stats, index=[0])

        if save_csv:
            if csv_file_path:
                summary_df.to_csv(csv_file_path, index=False)
            else:
                print("Unable to save Summary Statistics DataFrame to CSV.")

        return summary_df


    @staticmethod
    def query_dataset(
            query_engine, 
            eval_dataset
        ) -> Tuple[List[AgentChatResponse], List[float]]:
        """Query dataset and calculate elapsed times."""
        elapsed_times = []
        response_vectors = []
        for query in eval_dataset.queries:
            start_time = time.time()
            response_vectors.append(query_engine.query(query))
            end_time = time.time()

            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
        
        return response_vectors, elapsed_times
    
    @staticmethod
    async def aquery_dataset(
            query_engine, 
            eval_dataset
        ) -> Tuple[List[AgentChatResponse], List[float]]:
        """Query dataset asyncronously with query engine and track elapsed 
        times."""

        async def query_and_track_time(query):
            """Query the engine and track the elapsed time."""
            start_query_time = time.time()
            try:
                response = await query_engine.aquery(query)
            except Exception as e:
                print(f"ERROR: Unable to fetch response: {e}")
                response = AgentChatResponse(response="Unable to fetch response.")
            end_query_time = time.time()
            elapsed_query_time = end_query_time - start_query_time
            return response, elapsed_query_time
        
        # Run all queries concurrently
        tasks = [query_and_track_time(query) for query in eval_dataset.queries]
        response_times = await asyncio.gather(*tasks)
        response_vectors, elapsed_times = zip(*response_times)
        
        return response_vectors, elapsed_times
    
    @staticmethod
    async def achat_dataset(
            chat_engine, 
            eval_dataset
        ) -> Tuple[List[AgentChatResponse], List[float]]:
        """Query dataset asyncronously with chat engine and track elapsed 
        times."""

        async def query_and_track_time(query):
            """Query the engine and track the elapsed time."""
            start_query_time = time.time()
            try:
                response = await chat_engine.achat(query)
            except Exception as e:
                # TODO: print error to log as warning
                print(f"ERROR: Unable to fetch response: {e}")
                response = AgentChatResponse(
                    response="Unable to fetch response."
                )
            end_query_time = time.time()
            elapsed_query_time = end_query_time - start_query_time
            return response, elapsed_query_time
        
        # Run all queries concurrently
        tasks = [query_and_track_time(query) for query in eval_dataset.queries]
        response_times = await asyncio.gather(*tasks)
        response_vectors, elapsed_times = zip(*response_times)
        
        return response_vectors, elapsed_times
    
    @staticmethod
    async def _evaluate(
            eval_dataset,         
            response_vectors,
            llm: OpenAI = OpenAI(temperature=0, model="gpt-4")
        ):

        relevancy_evaluator = RelevancyEvaluator()
        faithfulness_evaluator = FaithfulnessEvaluator()
        correctness_evaluator = CorrectnessEvaluator(
            parser_function=custom_parser
        )
        semantic_evaluator = SemanticSimilarityEvaluator()

        eval_results = []
        print(f"Evaluating {len(response_vectors)} responses...")
        for i, response in enumerate(response_vectors):

            # NOTE: response is of type AgentChatResponse, which means that we 
            # also have "response.source_nodes", which we could use for further
            # evaluation

            # Measure if the query was actually answered by the response
            # (response + source nodes match the query)
            eval_relevancy = await relevancy_evaluator.aevaluate_response(
                query=eval_dataset.queries[i],
                response=response,
                llm=llm
            )
            # Measure if the response matches any source nodes
            eval_faithfulness = await faithfulness_evaluator.aevaluate_response(
                response=response,
                llm=llm
            )
            # Evaluates relevance and correctness of the response against a 
            # reference answer
            eval_correctness = await correctness_evaluator.aevaluate_response(
                query=eval_dataset.queries[i],
                response=response,
                reference=eval_dataset.answers[i],
                llm=llm
            )
            # Evaluates the quality of a question answering system via semantic
            # similarity (calculates the similarity score between embeddings of
            #  the generated answer and the reference answer)
            eval_semantic = await semantic_evaluator.aevaluate_response(
                response=response,
                reference=eval_dataset.answers[i],
                similarity_mode=SimilarityMode.DEFAULT, # Cosine distance
                similarity_threshold=DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
                llm=llm
            )

            eval_results.append({
                "relevancy": eval_relevancy,
                "faithfulness": eval_faithfulness,
                "correctness": eval_correctness,
                "semantic_similarity": eval_semantic
            })
        
        return eval_results

    @classmethod
    async def from_query_engine(
        cls,
        aquery_engine: RetrieverQueryEngine,
        llm: OpenAI = OpenAI(temperature=0, model="gpt-4"),
        **kwargs
    ) -> "Eval":
        """Evaluates the RAG system up to the Query Engine.
        
        The evaluation is perfmed over the reader, storage, index and query
        modules.
        """
        eval_dataset = cls.get_eval_dataset(**kwargs)
        print("Getting responses from query engine...")
        response_vectors, elapsed_times = await cls.aquery_dataset(
            aquery_engine, 
            eval_dataset
        )
        eval_results = await cls._evaluate(
            eval_dataset, 
            response_vectors,
            llm
        )
        return cls(
            results=eval_results,
            elapsed_times=elapsed_times
        )

    @classmethod
    async def from_chat_engine(
        cls,
        achat_engine: BaseChatEngine,
        llm: OpenAI = OpenAI(temperature=0, model="gpt-4"),
        **kwargs
    ) -> "Eval":
        """Evaluates the full RAG system, up to the Chat Engine.
        
        The evaluation is perfmed over the reader, storage, index, query and 
        chat modules.
        """
        eval_dataset = cls.get_eval_dataset(**kwargs)
        print("Getting responses from query engine...")
        response_vectors, elapsed_times = await cls.achat_dataset(
            achat_engine, 
            eval_dataset
        )
        eval_results = await cls._evaluate(
            eval_dataset, 
            response_vectors,
            llm
        )
        return cls(
            results=eval_results,
            elapsed_times=elapsed_times
        )
        


if __name__ == "__main__":

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    # reader = CustomMongoReader(uri=mongodb_uri)
    # documents = reader.load_data(**eval_mongo_reader_config)

    # # Generate and save dataset
    # if GENERATE_DATASET and len(documents) > 0:
    #     print("Generating evaluation dataset...") # TODO: change to log.info
    #     eval_dataset = EvalDataset.generate(
    #         documents=documents,
    #         num_questions_per_node=NUM_QUESTIONS_PER_NODE,
    #         show_progress=SHOW_PROGRESS
    #     )
    #     if SAVE_DATASET and DATASET_FILE_PATH != "":
    #         print("Saving evaluation dataset...") # TODO: change to log.info
    #         eval_dataset.save_json(
    #             file_path=DATASET_FILE_PATH
    #         )

    aquery_engine = load_async_query_engine()
    retriever = load_retriever()
    achat_engine = load_chat_engine(
        chat_mode=CHAT_MODE,
        chat_history=[], # No previous context during eval
        retriever=retriever,
        query_engine=aquery_engine,
        verbose=False
    )

    eval = asyncio.run(Eval.from_chat_engine(
        achat_engine=achat_engine,
        dataset_file_path=DATASET_FILE_PATH,
        llm=llm
    ))


    # From correctness_evaluator we are evaluating relevance and correctness of
    # the response against a reference answer
    # This means that we can measure:
    # -Exact Match (EM): The percentage of queries that are answered exactly correctly.
    # -Recall: The percentage of queries that are answered correctly, regardless of the number of answers returned.
    # -Precision: The percentage of queries that are answered correctly, divided by the number of answers returned.
    # -F1: The F1 score is the harmonic mean of precision and recall. It thus symmetrically represents both precision and recall in one metric, considering both false positives and false negatives.
    # We just need to decide which threashold is considered "correct" and "incorrect" from the returned scores
    # TODO: check this: https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54
    # TODO: resolve the issue with the custom_parser function above.



    # TODO: check that if some value are null, not sure how sum() will handle it.
    average_time = sum(eval.elapsed_times) / len(eval.elapsed_times)

    print("Avg. query time:", average_time)

    eval.get_df(
        save_csv=SAVE_EVAL_DATAFRAME, 
        csv_file_path=EVAL_DATAFRAME_FILE_PATH
    )

    eval.get_summary_statistics(
        save_csv=SAVE_SUMMARY_STATISTICS_DATAFRAME,
        csv_file_path=SUMMARY_STATISTICS_DATAFRAME_FILE_PATH
    )
