import hashlib
import os
import tempfile
import time
from pathlib import Path

import boto3
from botocore.config import Config
from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_BUCKET = os.environ["S3_BUCKET"]
QDRANT_URL = os.environ["QDRANT_URL"]
COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_SECRET_KEY"],
    config=Config(signature_version="s3v4"),
)
qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(MODEL_NAME)
converter = DocumentConverter()


def ensure_collection():
    names = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION not in names:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def chunk_text(text: str, max_chars: int = 2200, overlap: int = 250):
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        yield text[start:end]
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def table_to_markdown(table) -> str:
    # Docling reconstructs cells/rows/columns. Markdown preserves headers and row relations
    # much better for an LLM than flattening all cells into plain text.
    try:
        return table.export_to_markdown()
    except Exception:
        try:
            return table.export_to_dataframe().to_markdown(index=False)
        except Exception:
            return str(table)


def parse_document(pdf_path: Path):
    result = converter.convert(str(pdf_path))
    doc = result.document

    chunks = []
    # Main document export includes headings and paragraphs in reading order.
    markdown = doc.export_to_markdown()
    for i, text in enumerate(chunk_text(markdown)):
        chunks.append({"kind": "text", "content": text, "ordinal": i})

    # Index every table separately so queries about prices, dates, amounts, etc.
    # can retrieve the entire table instead of isolated cells.
    for i, table in enumerate(getattr(doc, "tables", [])):
        table_md = table_to_markdown(table)
        if table_md.strip():
            chunks.append({"kind": "table", "content": table_md, "ordinal": i})
    return chunks


def already_indexed(doc_hash: str) -> bool:
    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="sha256", match=models.MatchValue(value=doc_hash))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


def index_pdf(key: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(key).name
        s3.download_file(S3_BUCKET, key, str(path))
        digest = sha256_file(path)
        if already_indexed(digest):
            return

        chunks = parse_document(path)
        texts = [c["content"] for c in chunks]
        vectors = embedder.encode(texts, normalize_embeddings=True, batch_size=8).tolist()

        ids = []
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = int(hashlib.sha256(f"{digest}:{idx}".encode()).hexdigest()[:15], 16)
            ids.append(point_id)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": chunk["content"],
                        "title": Path(key).name,
                        "source": f"s3://{S3_BUCKET}/{key}",
                        "document_id": digest,
                        "sha256": digest,
                        "kind": chunk["kind"],
                        "ordinal": chunk["ordinal"],
                        "s3_bucket": S3_BUCKET,
                        "s3_key": key,
                    },
                )
            )
        qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)
        print(f"Indexed {key}: {len(points)} chunks ({sum(1 for c in chunks if c['kind']=='table')} tables)")


def scan_once():
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                try:
                    index_pdf(key)
                except Exception as exc:
                    print(f"ERROR {key}: {exc}")


def main():
    ensure_collection()
    while True:
        scan_once()
        time.sleep(30)


if __name__ == "__main__":
    main()
