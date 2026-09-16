from pathlib import Path
from uuid import uuid4
import re

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


MARKDOWN_PATH = Path("data/ocr/pages/Акт №91 от 28.04.2022_0.md")
MODEL_PATH = "models/bge-m3"
COLLECTION = "documents"
DOCUMENT_NAME = "акт №87 от 31.03.20.pdf"


def split_markdown(text: str, chunk_size: int = 1200, overlap: int = 200):
    # Сначала режем по крупным блокам Markdown
    blocks = re.split(r"\n(?=#{1,6}\s|\|)", text)

    chunks = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        # Небольшие блоки сохраняем целиком
        if len(block) <= chunk_size:
            chunks.append(block)
            continue

        # Большие режем с overlap
        start = 0

        while start < len(block):
            end = start + chunk_size
            chunk = block[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(block):
                break

            start = end - overlap

    return chunks


def main():
    print(f"Читаю: {MARKDOWN_PATH}")

    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")

    chunks = split_markdown(markdown)

    print(f"Создано chunks: {len(chunks)}")

    print("Загружаю BGE-M3...")

    model = SentenceTransformer(
        MODEL_PATH,
        local_files_only=True
    )

    vector_size = model.get_sentence_embedding_dimension()

    print(f"Размер embedding: {vector_size}")

    print("Создаю embeddings...")

    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    qdrant = QdrantClient(
        url="http://localhost:6333"
    )

    # Для MVP пересоздаём коллекцию
    if qdrant.collection_exists(COLLECTION):
        print(f"Удаляю старую коллекцию {COLLECTION}")
        qdrant.delete_collection(COLLECTION)

    print(f"Создаю коллекцию {COLLECTION}")

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    points = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector.tolist(),
                payload={
                    "document": DOCUMENT_NAME,
                    "chunk_index": index,
                    "text": chunk
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=points
    )

    print()
    print("Индексация завершена")
    print(f"Документ: {DOCUMENT_NAME}")
    print(f"Chunks: {len(chunks)}")
    print(f"Коллекция: {COLLECTION}")


if __name__ == "__main__":
    main()