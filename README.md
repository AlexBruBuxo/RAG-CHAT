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
Pull Qdrant:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/data/qdrant_storage:/qdrant/storage:z \
    --name qdrant \
    qdrant/qdrant
```
Stop Qdrant:
```bash
docker stop qdrant
```
Start Qdrant:
```bash
docker start qdrant
```

