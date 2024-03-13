import time
import asyncio
import pandas as pd
import numpy as np
import openai
import shutil
from typing import List, Dict, Tuple, Optional
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

from rag_chat.evaluation.dataset import EvalDataset, RetrievalEvalDataset
from rag_chat.query.query import load_async_query_engine, load_retriever
from rag_chat.agent.chat import load_chat_engine
from rag_chat.evaluation.metrics import MRR, HitRate, Recall
from rag_chat.evaluation.config import (
    DATASET_FILE_PATH,
    RETRIEVAL_DATASET_FILE_PATH,
    CHAT_MODE,
    SAVE_EVAL_DATAFRAME,
    EVAL_DATAFRAME_FILE_PATH,
    RETRIEVAL_EVAL_DATAFRAME_FILE_PATH,
    SAVE_SUMMARY_STATISTICS_DATAFRAME,
    SUMMARY_STATISTICS_DATAFRAME_FILE_PATH,
    INDEX_CONFIGURATION_FOLDER_PATH,
    INFERENCE_CONFIGURATION_FOLDER_PATH
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
            elapsed_times: List[float] = [],
            retrieval_results: List[Dict] = [],
            llm: OpenAI = OpenAI(temperature=0, model="gpt-4")
        ):
        self.results = results
        self.elapsed_times = elapsed_times
        self.retrieval_results = retrieval_results
        self.llm = llm
    
    @staticmethod
    def get_eval_dataset(
        dataset_file_path: str = "",
        retrieval_dataset_file_path: str = "",
    ) -> EvalDataset:
        """Load evaluation from json file."""
        print("Loading evaluation datasets...") # TODO: change to log.info
        eval_dataset = EvalDataset.from_json(
            file_path=dataset_file_path
        )
        retrieval_eval_dataset = RetrievalEvalDataset.from_json(
            file_path=retrieval_dataset_file_path
        )
        return eval_dataset, retrieval_eval_dataset

    def get_df(
            self,
            save_csv: bool = False, 
            csv_file_path: str = None,
            retrieval_csv_file_path: str = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # Eval Dataframe
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
        
        # Retrieval Eval Dataframe
        retrieval_data = []
        for result in self.retrieval_results:
            row = {
                "Hit_Rate": result["hit_rate"].score,
                "MRR": result["mrr"].score,
                "Recall": result["recall"].score
            }
            retrieval_data.append(row)
        retrieval_df = pd.DataFrame(retrieval_data)

        if save_csv:
            if retrieval_csv_file_path:
                retrieval_df.to_csv(retrieval_csv_file_path, index=False)
            else:
                print("Unable to save Retrieval Evaluation DataFrame to CSV.")

        return df, retrieval_df
    
    def get_summary_statistics(
            self,
            save_csv: bool = False, 
            csv_file_path: str = None
        ) -> pd.DataFrame:
        score_metrics = [
            "faithfulness",
            "correctness",
            "semantic_similarity"
        ]
        passing_metrics = [
            "relevancy",
            "faithfulness",
            "correctness",
            "semantic_similarity"
        ]
        summary_stats = {}

        # Eval Metrics

        for metric in score_metrics:
            values = []
            for result in self.results:
                if hasattr(result[metric], 'score'):
                    values.append(result[metric].score)
            if len(values) > 0:
                summary_stats[metric + "_score_mean"] = np.mean(values)
                summary_stats[metric + "_score_variance"] = np.var(values)
                summary_stats[metric + "_score_p90"] = np.percentile(values, 90)
        
        for metric in passing_metrics:
            values = []
            for result in self.results:
                if hasattr(result[metric], 'passing'):
                    # Convert "TRUE" and "FALSE" strings to boolean values
                    values.append(str(result[metric].passing).upper() == "TRUE")

            if len(values) > 0:
                summary_stats[metric + "_passing_rate"] = sum(values) / len(values) * 100 if values else 0

        summary_stats["avg_query_time"] = sum(self.elapsed_times) / len(self.elapsed_times)

        # Retrieval Eval Metrics

        hit_rates = []
        mrrs = []
        recalls = []
        for result in self.retrieval_results:
            hit_rates.append(result["hit_rate"].score)
            mrrs.append(result["mrr"].score)
            recalls.append(result["recall"].score)

        summary_stats["avg_hit_rate"] = sum(hit_rates) / len(hit_rates) * 100 if hit_rates else 0
        summary_stats["avg_mrr"] = sum(mrrs) / len(mrrs) * 100 if mrrs else 0
        summary_stats["avg_recall"] = sum(recalls) / len(recalls) * 100 if recalls else 0

        # Build DataFrame

        summary_df = pd.DataFrame(summary_stats, index=[0])

        if save_csv:
            if csv_file_path:
                summary_df.to_csv(csv_file_path, index=False)
            else:
                print("Unable to save Summary Statistics DataFrame to CSV.")

        return summary_df

    @staticmethod
    async def query_dataset(
            query_engine, 
            eval_dataset
        ) -> Tuple[List[AgentChatResponse], List[float]]:
        """Query dataset with query engine and track elapsed times."""
        # NOTE: due to constant RateLimitError limitations from OpenAI this is
        # temporarily transformed to synchronous
        response_vectors = []
        elapsed_times = []

        for query in eval_dataset.queries:
            start_query_time = time.time()
            try:
                response = await query_engine.aquery(query)
            except openai.RateLimitError as e:
                retry_after = int(e.body['message'].split("Please try again in ")[1].split("s.")[0].replace('.', '')) + 100 / 1000
                print(f"Rate limit reached. Waiting for {retry_after} seconds before retrying.") # TODO: print as warning
                await asyncio.sleep(retry_after)
                # Retry the query
                response = await query_engine.aquery(query)
            except Exception as e:
                print(f"ERROR: Unable to fetch response: {e}") # TODO: print as warning
                response = AgentChatResponse(response="Unable to fetch response.")

            end_query_time = time.time()
            elapsed_query_time = end_query_time - start_query_time

            response_vectors.append(response)
            elapsed_times.append(elapsed_query_time)
        
        return response_vectors, elapsed_times
    
    @staticmethod
    async def chat_dataset(
            chat_engine, 
            eval_dataset
        ) -> Tuple[List[AgentChatResponse], List[float]]:
        """Query dataset with chat engine and track elapsed times."""
        # NOTE: due to constant RateLimitError limitations from OpenAI this is
        # temporarily transformed to synchronous
        response_vectors = []
        elapsed_times = []

        for query in eval_dataset.queries:
            start_query_time = time.time()
            try:
                response = await chat_engine.achat(query)
            except openai.RateLimitError as e:
                retry_after = int(e.body['message'].split("Please try again in ")[1].split("s.")[0].replace('.', '')) + 100 / 1000
                print(f"Rate limit reached. Waiting for {retry_after} seconds before retrying.") # TODO: print as warning
                await asyncio.sleep(retry_after)
                # Retry the query
                response = await chat_engine.achat(query)
            except Exception as e:
                print(f"ERROR: Unable to fetch response: {e}") # TODO: print as warning
                response = AgentChatResponse(response="Unable to fetch response.")

            end_query_time = time.time()
            elapsed_query_time = end_query_time - start_query_time

            response_vectors.append(response)
            elapsed_times.append(elapsed_query_time)
        
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
            # the generated answer and the reference answer)
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

    @staticmethod
    def _retrieval_evaluate(
        retrieval_eval_dataset,         
        retrieval_response_vectors,
    ):
        hit_rate = HitRate() # Checks if any expected_id is in retrieved_nodes (1 or 0)
        mrr = MRR() # Checks the position of the expected_ids within the retrieved Nodes
        recall = Recall() # Checks the proportion of relevant Nodes that were successfully retrieved
        
        retrieval_eval_results = []
        print(f"Evaluating {len(retrieval_response_vectors)} retrieval responses...")
        for i, response in enumerate(retrieval_response_vectors):
            retrieved_ids = [node.id_ for node in response.source_nodes]

            hit_rate_result = hit_rate.compute(
                expected_ids=retrieval_eval_dataset.expected_ids[i], 
                retrieved_ids=retrieved_ids
            )
            mrr_result = mrr.compute(
                expected_ids=retrieval_eval_dataset.expected_ids[i], 
                retrieved_ids=retrieved_ids
            )
            recall_result = recall.compute(
                expected_ids=retrieval_eval_dataset.expected_ids[i], 
                retrieved_ids=retrieved_ids
            )
        
            retrieval_eval_results.append({
                "hit_rate": hit_rate_result,
                "mrr": mrr_result,
                "recall": recall_result
            })
    
        return retrieval_eval_results

    
    def save_conf_files(
            self,
            index_conf_folder_path: str,
            inference_conf_folder_path: str
        ) -> None:
        try:
            shutil.copy("conf/index_conf.yaml", index_conf_folder_path)
            shutil.copy("conf/inference_conf.yaml", inference_conf_folder_path)
        except Exception as e:
            print(f"An error occurred while copying the configuration: {e}")

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
        eval_dataset, retrieval_eval_dataset = cls.get_eval_dataset(**kwargs)
        print("Getting responses from query engine...")
        response_vectors, elapsed_times = await cls.query_dataset(
            aquery_engine, 
            eval_dataset
        )
        eval_results = await cls._evaluate(
            eval_dataset, 
            response_vectors,
            llm
        )
        print("Getting nodes from retriever...")
        retrieval_response_vectors, _ = await cls.query_dataset(
            aquery_engine,
            retrieval_eval_dataset
        )
        retrieval_eval_results = cls._retrieval_evaluate(
            retrieval_eval_dataset, 
            retrieval_response_vectors
        )
        return cls(
            results=eval_results,
            elapsed_times=elapsed_times,
            retrieval_results=retrieval_eval_results
        )

    @classmethod
    async def from_chat_engine(
        cls,
        achat_engine: BaseChatEngine,
        aquery_engine: RetrieverQueryEngine,
        llm: OpenAI = OpenAI(temperature=0, model="gpt-4"),
        **kwargs
    ) -> "Eval":
        """Evaluates the full RAG system, up to the Chat Engine.
        
        The evaluation is perfmed over the reader, storage, index, query and 
        chat modules.
        """
        eval_dataset, retrieval_eval_dataset = cls.get_eval_dataset(**kwargs)
        print("Getting responses from chat engine...")
        response_vectors, elapsed_times = await cls.chat_dataset(
            achat_engine, 
            eval_dataset
        )
        eval_results = await cls._evaluate(
            eval_dataset, 
            response_vectors,
            llm
        )
        print("Getting nodes from retriever...")
        retrieval_response_vectors, _ = await cls.query_dataset(
            aquery_engine,
            retrieval_eval_dataset
        )
        retrieval_eval_results = cls._retrieval_evaluate(
            retrieval_eval_dataset, 
            retrieval_response_vectors
        )
        return cls(
            results=eval_results,
            elapsed_times=elapsed_times,
            retrieval_results=retrieval_eval_results
        )



if __name__ == "__main__":

    from llama_index.llms import ChatMessage, MessageRole

    llm = OpenAI(temperature=0, model="gpt-4")
    aquery_engine = load_async_query_engine()
    retriever = load_retriever()

    achat_engine = load_chat_engine(
        chat_mode=CHAT_MODE,
        chat_history=[],  # No previous context during eval
        retriever=retriever,
        query_engine=aquery_engine,
        verbose=False
    )

    eval = asyncio.run(Eval.from_chat_engine(
        achat_engine=achat_engine,
        aquery_engine=aquery_engine,
        dataset_file_path=DATASET_FILE_PATH,
        retrieval_dataset_file_path=RETRIEVAL_DATASET_FILE_PATH,
        llm=llm
    ))

    eval.get_df(
        save_csv=SAVE_EVAL_DATAFRAME, 
        csv_file_path=EVAL_DATAFRAME_FILE_PATH,
        retrieval_csv_file_path=RETRIEVAL_EVAL_DATAFRAME_FILE_PATH
    )

    eval.get_summary_statistics(
        save_csv=SAVE_SUMMARY_STATISTICS_DATAFRAME,
        csv_file_path=SUMMARY_STATISTICS_DATAFRAME_FILE_PATH
    )

    eval.save_conf_files(
        index_conf_folder_path=INDEX_CONFIGURATION_FOLDER_PATH,
        inference_conf_folder_path=INFERENCE_CONFIGURATION_FOLDER_PATH
    )
