# RAG-Chat

This project explores the potential of RAG systems in the context of an online product store, aiming to enhance the accuracy, relevance, and reliability of LLM-generated responses using a subset of the [Walmart Product Data 2019 dataset](https://www.kaggle.com/datasets/promptcloud/walmart-product-data-2019). The research adopts an incremental approach, systematically building and validating each component of the RAG system prior to integration. The methodology involves a comprehensive process, including data collection and ingestion, the implementation of retriever and postprocessor modules, and the exploration of prompting techniques such as Condense Plus Context, Condense, and ReAct (refer to [LlamaIndex](https://www.llamaindex.ai/)). These elements, along with a user-friendly interface, are designed to facilitate natural and effective user interactions.

The evaluation process enables iterative experimentation and optimization, refining various metrics to assess context retrieval and text generation. The results demonstrate a 100\% score for Hit Rate and MRR, with 97.79\% Recall for retrieval metrics, and an average score of 82.24\% across generation metrics, including Relevancy, Faithfulness, Correctness, and Semantic Similarity. These findings significantly contribute to the creation of more reliable and accurate LLM-powered applications in the e-commerce domain. Future work could focus on scaling this research to develop production-ready RAG systems. 

## Development

### Poetry

Run scripts with:
```bash
poetry run python <script.py>
```
or activate the environment:
```bash
poetry shell
```

### Qdrant

Pull Qdrant:
```bash
docker pull qdrant/qdrant
```
Run Qdrant:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/data/qdrant_storage:/qdrant/storage:z \
    --name qdrant \
    qdrant/qdrant
```
Start Qdrant:
```bash
docker start qdrant
```
URL: http://localhost:6333/dashboard
Stop Qdrant:
```bash
docker stop qdrant
```
Remove Qdrant:
```bash
docker rm qdrant
```

### Frontend/Backend

Run API:
```bash
poetry run python rag_chat/api.py
```

Run frontend:
```bash
cd frontend
python -m http.server
```

