# -*- coding: utf-8 -*-

from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
import argparse
import re

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

from sentence_transformers import SentenceTransformer


MODEL_PATH = "models/bge-m3"
COLLECTION = "documents"
QDRANT_URL = "http://localhost:6333"

CHUNK_SIZE = 1200
OVERLAP = 200


# ------------------------------------------------------
# Получаем номер страницы из имени файла
#
# Акт №87 от 31.03.20_0.md -> 0
# Акт №87 от 31.03.20_1.md -> 1
# ------------------------------------------------------

def get_page_index(markdown_path: Path) -> int:
    match = re.search(
        r"_(\d+)$",
        markdown_path.stem,
    )

    if not match:
        return 0

    return int(
        match.group(1)
    )


# ------------------------------------------------------
# Chunking Markdown
# ------------------------------------------------------

def split_markdown(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
):
    blocks = re.split(
        r"\n(?=#{1,6}\s|\|)",
        text,
    )

    chunks = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        if len(block) <= chunk_size:
            chunks.append(block)
            continue

        start = 0

        while start < len(block):
            end = min(
                start + chunk_size,
                len(block),
            )

            chunk = block[
                start:end
            ].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(block):
                break

            start = end - overlap

    return chunks


# ------------------------------------------------------
# Удаляем старую версию конкретного документа
# ------------------------------------------------------

def delete_existing_document(
    client: QdrantClient,
    document_name: str,
):
    if not client.collection_exists(
        COLLECTION
    ):
        return

    print(
        f"Удаляю старые chunks документа "
        f"из Qdrant: {document_name}"
    )

    client.delete(
        collection_name=COLLECTION,

        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="document",
                        match=MatchValue(
                            value=document_name
                        ),
                    )
                ]
            )
        ),

        wait=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Создание embeddings для всего OCR-документа "
            "и загрузка в Qdrant"
        )
    )

    parser.add_argument(
        "document",
        help=(
            "Путь к каталогу документа, например: "
            "data/documents/Акт №87 от 31.03.20"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # Каталог документа
    # ------------------------------------------------------

    document_dir = Path(
        args.document
    )

    if not document_dir.exists():
        raise FileNotFoundError(
            f"Каталог документа не найден: "
            f"{document_dir}"
        )

    if not document_dir.is_dir():
        raise ValueError(
            f"Ожидался каталог документа: "
            f"{document_dir}"
        )

    document_name = (
        document_dir.name
    )

    pages_dir = (
        document_dir /
        "pages"
    )

    if not pages_dir.exists():
        raise FileNotFoundError(
            f"Каталог pages не найден: "
            f"{pages_dir}"
        )

    # ------------------------------------------------------
    # Находим Markdown страницы
    #
    # ВАЖНО:
    # glob("*.md") смотрит только непосредственно
    # в pages/.
    #
    # pages/imgs вообще не затрагивается.
    # ------------------------------------------------------

    markdown_files = list(
        pages_dir.glob("*.md")
    )

    if not markdown_files:
        raise FileNotFoundError(
            f"В каталоге нет Markdown-файлов: "
            f"{pages_dir}"
        )

    # Сортируем по номеру страницы.
    #
    # _0.md
    # _1.md
    # _2.md
    # ...
    # _10.md
    #
    # а не:
    #
    # _0.md
    # _1.md
    # _10.md
    # _2.md

    markdown_files.sort(
        key=get_page_index
    )

    print("=" * 80)
    print(
        f"Документ: {document_name}"
    )
    print(
        f"Каталог:  {document_dir}"
    )
    print(
        f"Страниц:  {len(markdown_files)}"
    )
    print("=" * 80)

    # ------------------------------------------------------
    # Собираем chunks со всех страниц
    # ------------------------------------------------------

    document_chunks = []

    global_chunk_index = 0

    for markdown_path in markdown_files:
        page_index = get_page_index(
            markdown_path
        )

        # Для человека страницы считаем с 1.
        page_number = (
            page_index + 1
        )

        print(
            f"Читаю страницу {page_number}: "
            f"{markdown_path.name}"
        )

        markdown = (
            markdown_path.read_text(
                encoding="utf-8"
            )
        )

        page_chunks = split_markdown(
            markdown
        )

        print(
            f"  chunks: {len(page_chunks)}"
        )

        for page_chunk_index, chunk in enumerate(
            page_chunks
        ):
            document_chunks.append(
                {
                    "chunk_index":
                        global_chunk_index,

                    "page":
                        page_number,

                    "page_index":
                        page_index,

                    "page_chunk_index":
                        page_chunk_index,

                    "page_file":
                        markdown_path.name,

                    "text":
                        chunk,
                }
            )

            global_chunk_index += 1

    if not document_chunks:
        raise RuntimeError(
            "Не удалось создать ни одного chunk"
        )

    print()
    print(
        f"Всего chunks документа: "
        f"{len(document_chunks)}"
    )

    # ------------------------------------------------------
    # BGE-M3
    # ------------------------------------------------------

    print()
    print(
        "Загружаю BGE-M3..."
    )

    model = SentenceTransformer(
        MODEL_PATH,
        local_files_only=True,
    )

    vector_size = (
        model
        .get_sentence_embedding_dimension()
    )

    print(
        f"Размер embedding: "
        f"{vector_size}"
    )

    # ------------------------------------------------------
    # Текст для embedding
    # ------------------------------------------------------

    embedding_texts = []

    for chunk in document_chunks:
        embedding_texts.append(
            f"Документ: {document_name}\n"
            f"Страница: {chunk['page']}\n\n"
            f"{chunk['text']}"
        )

    print()
    print(
        "Создаю embeddings..."
    )

    vectors = model.encode(
        embedding_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # ------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------

    client = QdrantClient(
        url=QDRANT_URL
    )

    # ------------------------------------------------------
    # Создаём коллекцию только если её ещё нет
    # ------------------------------------------------------

    if not client.collection_exists(
        COLLECTION
    ):
        print(
            f"Создаю коллекцию "
            f"{COLLECTION}"
        )

        client.create_collection(
            collection_name=COLLECTION,

            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    # ------------------------------------------------------
    # Удаляем старые chunks ТОЛЬКО этого документа
    # ------------------------------------------------------

    delete_existing_document(
        client,
        document_name,
    )

    # ------------------------------------------------------
    # Points
    # ------------------------------------------------------

    points = []

    for chunk, vector in zip(
        document_chunks,
        vectors,
    ):
        chunk_index = (
            chunk["chunk_index"]
        )

        # UUID стабильный внутри документа.
        point_id = str(
            uuid5(
                NAMESPACE_URL,
                (
                    f"{document_name}:"
                    f"{chunk_index}"
                ),
            )
        )

        points.append(
            PointStruct(
                id=point_id,

                vector=vector.tolist(),

                payload={
                    "document":
                        document_name,

                    "page":
                        chunk["page"],

                    "page_index":
                        chunk["page_index"],

                    "page_file":
                        chunk["page_file"],

                    "page_chunk_index":
                        chunk[
                            "page_chunk_index"
                        ],

                    "chunk_index":
                        chunk_index,

                    "text":
                        chunk["text"],
                },
            )
        )

    # ------------------------------------------------------
    # Upsert
    # ------------------------------------------------------

    print()
    print(
        f"Записываю "
        f"{len(points)} chunks "
        f"в Qdrant..."
    )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )

    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("Индексация завершена")
    print("=" * 80)

    print(
        f"Документ: {document_name}"
    )

    print(
        f"Страниц:  {len(markdown_files)}"
    )

    print(
        f"Chunks:   {len(document_chunks)}"
    )

    info = client.get_collection(
        COLLECTION
    )

    print(
        f"Всего points в Qdrant: "
        f"{info.points_count}"
    )


if __name__ == "__main__":
    main()