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
