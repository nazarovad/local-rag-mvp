# Local RAG MVP: Open WebUI + S3 + Docling + Qdrant + vLLM

MVP for a fully local/air-gapped RAG system. PDF files are read from S3, parsed with Docling, text and reconstructed tables are embedded separately, and vectors are stored in Qdrant. Open WebUI is the browser UI and vLLM serves the local LLM over an OpenAI-compatible API.

## Components

- Open WebUI — browser chat UI
- vLLM — local OpenAI-compatible LLM endpoint
- S3/MinIO — source PDF storage
- Docling — PDF layout parsing, OCR-capable pipeline and table reconstruction
- BGE-M3 — multilingual embeddings
- Qdrant — vector store
- rag-worker — ingestion from S3
- rag-api — simple retrieval/debug API
- PostgreSQL — reserved for document status, ACL and ingestion history in the next iteration

## Why tables are indexed separately

Docling reconstructs table cell/row/column structure. The worker exports each detected table to Markdown and stores it as a dedicated `kind=table` vector chunk. This avoids a common RAG failure mode where a row is split from its column headers. The complete document Markdown is also chunked, so narrative context around the table remains searchable.

For production, the next improvement should be contextual table chunks such as:

`<document title> -> <section title> -> <page> -> <table caption> -> <markdown table>`

and large tables should be split by row groups while repeating column headers.

## Start

1. Copy `.env.example` to `.env`.
2. Put all model files in `./models`. In an air-gapped environment, no service should download models at runtime.
3. Set `VLLM_MODEL` to the local model directory.
4. Create the S3 bucket `documents` (or change `S3_BUCKET`).
5. Run `docker compose up --build`.
6. Open `http://localhost:3000`.

## Open WebUI connection

Open WebUI talks to `http://vllm:8000/v1` using the OpenAI-compatible API.

For the fastest MVP RAG integration, configure Qdrant in Open WebUI as an External Knowledge Source and map:

- content -> `content`
- title -> `title`
- source -> `source`
- document ID -> `document_id`
- metadata -> payload/metadata as supported by your Open WebUI version

Important: Open WebUI must use exactly the same embedding model and vector dimension as the worker. If the worker creates BGE-M3 vectors, the query embedding must also be BGE-M3.

## Test retrieval

```bash
curl -X POST http://localhost:8081/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Какова сумма договора и сроки оплаты?","limit":8}'
```

A table match will return `"kind":"table"` and the table itself in Markdown.

## MVP limitations / next iteration

- Replace polling with S3 events -> Artemis/RabbitMQ.
- Add PostgreSQL ingestion state and errors.
- Add document versions and delete/reindex behavior.
- Add page/section/table-caption metadata from Docling provenance.
- Add ACL metadata and Qdrant filters per user/group.
- Add hybrid BM25+dense search and a reranker.
- Pin exact Docker/package versions before closed-contour deployment.
- Put Qdrant, vLLM and internal APIs on non-exposed networks; expose only NGINX/Open WebUI.

---
# Настройка проекта

Зависимости
```shell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r scripts/requirements.txt
python -m pip install -r rag-api/requirements.txt
```

Модели
```shell
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3-8B-AWQ --local-dir models/Qwen3-8B-AWQ
hf download BAAI/bge-m3 --local-dir models/bge-m3
```

Проверка работает ли GPU (только NVIDIA)
```shell
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```


---
1. Для работы нужно запустить модель Qwen3-8B
```shell
docker run --rm `
  --gpus all `
  --ipc=host `
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 `
  -p 8000:8000 `
  -v "${PWD}\models\Qwen3-8B-AWQ:/models/Qwen3-8B-AWQ:ro" `
  vllm/vllm-openai:latest `
  /models/Qwen3-8B-AWQ `
  --served-model-name qwen3-8b `
  --quantization awq `
  --max-model-len 4096 `
  --gpu-memory-utilization 0.85
```
`-e VLLM_WSL2_ENABLE_PIN_MEMORY=1` использовать при WSL2-ошибки с UVA

2. Запустить квадрант
```shell
docker run -d `
  --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v qdrant-storage:/qdrant/storage `
  qdrant/qdrant:latest
```

Коллекции находятся в `http://localhost:6333/collections/documents`

3. Запустить webUI
```shell
docker run -d `
  -p 3000:8080 `
  --add-host=host.docker.internal:host-gateway `
  -v open-webui:/app/backend/data `
  --name open-webui `
  --restart unless-stopped `
  ghcr.io/open-webui/open-webui:main
```

Open webUI `http://localhost:3000`

4. Запустить rag-api
```shell
uvicorn rag-api.app.main:app `
  --host 0.0.0.0 `
  --port 8001 `
  --reload
```
---
## Скрипты
1. ocr_extract.py - распознает текст
2. ingest_ocr.py - эмбеддинг

```shell
python .\scripts\ocr_extract.py "data/акт №12.pdf"
python .\scripts\ingest_ocr.py "data/documents/акт №12"   
```


