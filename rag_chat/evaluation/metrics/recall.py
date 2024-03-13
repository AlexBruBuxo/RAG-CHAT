from typing import List, Optional
from llama_index.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
    RetrievalMetricResult,
)



class Recall(BaseRetrievalMetric):
    """Recall metric."""

    metric_name: str = "recall"

    def compute(
        self,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
    ) -> RetrievalMetricResult:
        """Compute metric."""
        if retrieved_ids is None or expected_ids is None:
            raise ValueError("Retrieved ids and expected ids must be provided")
        
        num_relevant_retrieved = sum(1 for id in retrieved_ids if id in expected_ids)
        recall = num_relevant_retrieved / len(expected_ids) if len(expected_ids) > 0 else 0.0
        
        return RetrievalMetricResult(
            score=recall,
        )