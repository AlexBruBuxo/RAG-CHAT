# RAG-Chat

...

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

