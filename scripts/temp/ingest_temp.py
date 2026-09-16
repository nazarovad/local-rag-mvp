from pathlib import Path
from uuid import uuid4

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


PDF_PATH = Path("data/акт №87 от 31.03.20.pdf")
MODEL_PATH = "models/bge-m3"
COLLECTION = "documents"


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def main():
    pipeline_options = PdfPipelineOptions()

    # Включаем OCR
    pipeline_options.do_ocr = True

    # OCR даже если в PDF уже есть текстовый слой
    pipeline_options.ocr_options.force_full_page_ocr = True

    # Распознавание таблиц
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    print(f"Обрабатываю: {PDF_PATH}")

    result = converter.convert(PDF_PATH)
    document = result.document

    markdown = document.export_to_markdown()

    Path("data/parsed.md").write_text(
        markdown,
        encoding="utf-8"
    )

    print(markdown[:3000])

    chunks = split_text(markdown)

    model = SentenceTransformer(
        MODEL_PATH,
        local_files_only=True
    )

    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    qdrant = QdrantClient(
        url="http://localhost:6333"
    )

    vector_size = model.get_sentence_embedding_dimension()

    # Для теста пересоздаём коллекцию
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)

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
                    "document": PDF_PATH.name,
                    "chunk_index": index,
                    "text": chunk
                }
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=points
    )

    print(f"Готово. Индексировано chunks: {len(chunks)}")


if __name__ == "__main__":
    main()