from contextlib import asynccontextmanager
from typing import Any
import re
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL_PATH = "models/bge-m3"

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "documents"

VLLM_URL = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL = "qwen3-8b"

RAG_MODEL_NAME = "contracts-rag"

SEARCH_LIMIT = 8

MAX_CONTEXT_CHARS = 12000
MAX_SUMMARY_CONTEXT_CHARS = 18000


# ============================================================
# GLOBAL SERVICES
# ============================================================

embedding_model: SentenceTransformer | None = None
qdrant: QdrantClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    global qdrant

    print("Загрузка BGE-M3...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_PATH,
        local_files_only=True,
        device="cpu",
    )

    print("Подключение к Qdrant...")

    qdrant = QdrantClient(
        url=QDRANT_URL,
    )

    print("RAG API готов.")

    yield


app = FastAPI(
    title="Local Contracts RAG API",
    version="0.3.0",
    lifespan=lifespan,
)


# ============================================================
# API MODELS
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]

    temperature: float = 0.2
    max_tokens: int = 1000
    stream: bool = False


class SearchRequest(BaseModel):
    query: str
    limit: int = SEARCH_LIMIT


# ============================================================
# MESSAGE HELPERS
# ============================================================

def find_last_user_message(
    messages: list[ChatMessage],
) -> str:

    for message in reversed(messages):
        if message.role == "user":
            return message.content.strip()

    raise HTTPException(
        status_code=400,
        detail="Не найдено сообщение пользователя",
    )


# ============================================================
# DOCUMENT DETECTION
# ============================================================

def detect_document_number(
    question: str,
) -> str | None:

    patterns = [
        r"\bакт(?:е|а|у|ом)?\s*№\s*(\d+)",
        r"\bакт(?:е|а|у|ом)?\s*[nN]\s*\.?\s*(\d+)",
        r"\bакт(?:е|а|у|ом)?\s+номер\s+(\d+)",
        r"\bакт(?:е|а|у|ом)?\s+(\d+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return None


def is_summary_request(
    question: str,
) -> bool:

    text = question.lower()

    phrases = [
        "опиши",
        "расскажи",
        "что написано",
        "о чем",
        "о чём",
        "содержание",
        "краткое содержание",
        "резюме",
        "суммариз",
        "обобщи",
        "перескажи",
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


# ============================================================
# DOCUMENT DISCOVERY
# ============================================================

def get_all_document_names() -> list[str]:

    if qdrant is None:
        raise RuntimeError(
            "Qdrant не инициализирован"
        )

    result: set[str] = set()

    offset = None

    while True:

        points, next_offset = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:

            payload = point.payload or {}

            document = payload.get("document")

            if document:
                result.add(str(document))

        if next_offset is None:
            break

        offset = next_offset

    return sorted(result)


def resolve_document_name(
    document_number: str,
) -> str | None:

    documents = get_all_document_names()

    patterns = [
        rf"\bакт\s*№\s*{re.escape(document_number)}\b",
        rf"\bакт\s*[nN]\s*{re.escape(document_number)}\b",
        rf"\bакт\s+{re.escape(document_number)}\b",
    ]

    matches = []

    for document in documents:

        for pattern in patterns:

            if re.search(
                pattern,
                document,
                re.IGNORECASE,
            ):
                matches.append(document)
                break

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(
            f"WARNING: найдено несколько документов "
            f"с номером {document_number}: {matches}"
        )

        return matches[0]

    return None


def make_document_filter(
    document_name: str,
) -> Filter:

    return Filter(
        must=[
            FieldCondition(
                key="document",
                match=MatchValue(
                    value=document_name
                ),
            )
        ]
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question: str,
    limit: int = SEARCH_LIMIT,
    document_name: str | None = None,
) -> list[dict[str, Any]]:

    if embedding_model is None:
        raise RuntimeError(
            "Embedding model не инициализирован"
        )

    if qdrant is None:
        raise RuntimeError(
            "Qdrant не инициализирован"
        )

    vector = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    query_filter = None

    if document_name:
        query_filter = make_document_filter(
            document_name
        )

    result = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector.tolist(),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    items = []

    for point in result.points:

        payload = point.payload or {}

        items.append(
            {
                "score": float(point.score),
                "document": payload.get(
                    "document",
                    "Неизвестный документ",
                ),
                "chunk_index": payload.get(
                    "chunk_index",
                    -1,
                ),
                "text": payload.get(
                    "text",
                    "",
                ),
            }
        )

    return items


# ============================================================
# FULL DOCUMENT LOADING
# ============================================================

def load_document_chunks(
    document_name: str,
) -> list[dict[str, Any]]:

    if qdrant is None:
        raise RuntimeError(
            "Qdrant не инициализирован"
        )

    query_filter = make_document_filter(
        document_name
    )

    result = []

    offset = None

    while True:

        points, next_offset = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=query_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:

            payload = point.payload or {}

            result.append(
                {
                    "score": 1.0,
                    "document": payload.get(
                        "document",
                        document_name,
                    ),
                    "chunk_index": payload.get(
                        "chunk_index",
                        -1,
                    ),
                    "text": payload.get(
                        "text",
                        "",
                    ),
                }
            )

        if next_offset is None:
            break

        offset = next_offset

    result.sort(
        key=lambda x: x.get(
            "chunk_index",
            -1,
        )
    )

    return result


# ============================================================
# CONTEXT
# ============================================================

def build_context(
    documents: list[dict[str, Any]],
    max_chars: int,
) -> str:

    blocks = []
    current_size = 0

    for index, document in enumerate(
        documents,
        start=1,
    ):

        text = str(
            document.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        block = (
            f"[Фрагмент {index}]\n"
            f"Документ: {document['document']}\n"
            f"Chunk: {document['chunk_index']}\n\n"
            f"{text}"
        )

        if (
            current_size + len(block)
            > max_chars
        ):
            break

        blocks.append(block)

        current_size += len(block)

    return "\n\n---\n\n".join(
        blocks
    )


def get_source_documents(
    documents: list[dict[str, Any]],
) -> list[str]:

    sources = []

    for document in documents:

        name = document.get(
            "document"
        )

        if (
            name
            and name not in sources
        ):
            sources.append(name)

    return sources


# ============================================================
# ANSWER CLEANUP
# ============================================================

def remove_generated_sources(
    answer: str,
) -> str:

    """
    Qwen иногда самостоятельно пишет:
    'Использованные документы'

    Мы это удаляем, потому что источники
    должен контролировать сам RAG API.
    """

    patterns = [
        r"\n+---\s*\n+\**Использованные документы:\**.*$",
        r"\n+\**Использованные документы:\**.*$",
        r"\n+\**Источники:\**.*$",
    ]

    result = answer

    for pattern in patterns:

        result = re.sub(
            pattern,
            "",
            result,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

    return result.strip()


# ============================================================
# PROMPTS
# ============================================================

def build_system_prompt(
    context: str,
    document_name: str | None,
    summary_mode: bool,
) -> str:

    document_instruction = ""

    if document_name:

        document_instruction = f"""
Пользователь спрашивает о конкретном документе:

{document_name}

Используй информацию ТОЛЬКО из этого документа.
Не используй информацию из других документов.
"""

    if summary_mode:

        task_instruction = """
Пользователь просит описать документ.

Сформируй связное краткое содержание.

Опиши:

- что это за документ;
- кто является сторонами;
- какие работы или услуги указаны;
- существенные даты;
- суммы;
- НДС;
- номера договоров;
- важные условия;
- наличие претензий сторон.

Указывай только то, что реально присутствует
в предоставленном контексте.
"""

    else:

        task_instruction = """
Ответь непосредственно на вопрос пользователя.
"""

    return f"""
Ты — корпоративный помощник по документам.

КРИТИЧЕСКОЕ ПРАВИЛО:

Единственным источником истины является блок
КОНТЕКСТ, расположенный ниже.

Запрещено использовать собственные знания модели
для дополнения документа.

Не используй сведения из предыдущих ответов.

Не придумывай информацию.

Перед использованием любого значения убедись,
что оно присутствует в КОНТЕКСТЕ.

Особенно это касается:

- денежных сумм;
- НДС;
- дат;
- фамилий;
- названий организаций;
- номеров договоров;
- номеров актов;
- услуг;
- оборудования.

Если информация отсутствует в контексте,
не упоминай её.

Если ответа на вопрос нет, скажи:

"В предоставленных документах информация не найдена."

Не создавай раздел "Источники" или
"Использованные документы".
Источники будут добавлены системой отдельно.

Отвечай на русском языке.

{document_instruction}

{task_instruction}

КОНТЕКСТ:

{context}
""".strip()


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "embedding_model": "bge-m3",
        "qdrant_collection": (
            QDRANT_COLLECTION
        ),
        "llm": VLLM_MODEL,
    }


# ============================================================
# OPENAI MODELS
# ============================================================

@app.get("/v1/models")
def models():

    return {
        "object": "list",
        "data": [
            {
                "id": RAG_MODEL_NAME,
                "object": "model",
                "created": int(
                    time.time()
                ),
                "owned_by": "local-rag",
            }
        ],
    }


# ============================================================
# DEBUG: DOCUMENT LIST
# ============================================================

@app.get("/debug/documents")
def debug_documents():

    documents = get_all_document_names()

    return {
        "count": len(documents),
        "documents": documents,
    }


# ============================================================
# DEBUG: SINGLE DOCUMENT
# ============================================================

@app.get(
    "/debug/document/{document_number}"
)
def debug_document(
    document_number: str,
):

    document_name = resolve_document_name(
        document_number
    )

    if document_name is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Документ №"
                f"{document_number} "
                f"не найден"
            ),
        )

    chunks = load_document_chunks(
        document_name
    )

    return {
        "document": document_name,
        "chunks_count": len(chunks),
        "chunks": chunks,
    }


# ============================================================
# SEARCH DEBUG
# ============================================================

@app.post("/search")
def search(
    request: SearchRequest,
):

    document_number = (
        detect_document_number(
            request.query
        )
    )

    document_name = None

    if document_number:

        document_name = (
            resolve_document_name(
                document_number
            )
        )

        if document_name is None:

            return {
                "query": request.query,
                "document_number": (
                    document_number
                ),
                "document": None,
                "results": [],
            }

    documents = semantic_search(
        question=request.query,
        limit=request.limit,
        document_name=document_name,
    )

    return {
        "query": request.query,
        "document_number": (
            document_number
        ),
        "document": document_name,
        "results": documents,
    }


# ============================================================
# OPENAI CHAT
# ============================================================

@app.post("/v1/chat/completions")
async def chat(
    request: ChatCompletionRequest,
):

    question = find_last_user_message(
        request.messages
    )

    print()
    print("=" * 80)
    print(f"QUESTION: {question}")

    # --------------------------------------------------------
    # DOCUMENT DETECTION
    # --------------------------------------------------------

    document_number = (
        detect_document_number(
            question
        )
    )

    document_name = None

    if document_number:

        document_name = (
            resolve_document_name(
                document_number
            )
        )

        print(
            f"document_number="
            f"{document_number}"
        )

        print(
            f"document_name="
            f"{document_name}"
        )

        if document_name is None:

            return {
                "id": "chatcmpl-rag",
                "object": (
                    "chat.completion"
                ),
                "created": int(
                    time.time()
                ),
                "model": RAG_MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": (
                                "assistant"
                            ),
                            "content": (
                                f"Акт №"
                                f"{document_number} "
                                f"не найден."
                            ),
                        },
                        "finish_reason": (
                            "stop"
                        ),
                    }
                ],
            }

    # --------------------------------------------------------
    # RETRIEVAL MODE
    # --------------------------------------------------------

    summary_mode = (
        document_name is not None
        and is_summary_request(
            question
        )
    )

    if summary_mode:

        print(
            "RETRIEVAL MODE: "
            "FULL DOCUMENT"
        )

        documents = (
            load_document_chunks(
                document_name
            )
        )

        context = build_context(
            documents,
            MAX_SUMMARY_CONTEXT_CHARS,
        )

    else:

        print(
            "RETRIEVAL MODE: "
            "SEMANTIC"
        )

        documents = semantic_search(
            question=question,
            limit=SEARCH_LIMIT,
            document_name=document_name,
        )

        context = build_context(
            documents,
            MAX_CONTEXT_CHARS,
        )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print(
        f"chunks={len(documents)}"
    )

    for doc in documents:

        print(
            f"document="
            f"{doc['document']} "
            f"chunk="
            f"{doc['chunk_index']} "
            f"score="
            f"{doc['score']:.4f}"
        )

    # --------------------------------------------------------
    # EMPTY CONTEXT
    # --------------------------------------------------------

    if not context:

        return {
            "id": "chatcmpl-rag",
            "object": "chat.completion",
            "created": int(
                time.time()
            ),
            "model": RAG_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": (
                            "assistant"
                        ),
                        "content": (
                            "В предоставленных "
                            "документах информация "
                            "не найдена."
                        ),
                    },
                    "finish_reason": (
                        "stop"
                    ),
                }
            ],
        }

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = (
        build_system_prompt(
            context=context,
            document_name=(
                document_name
            ),
            summary_mode=(
                summary_mode
            ),
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Не передаём старую историю Open WebUI.
    #
    # Qwen получает только:
    #
    # system + current question
    #
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # --------------------------------------------------------
    # VLLM
    # --------------------------------------------------------

    payload = {
        "model": VLLM_MODEL,
        "messages": messages,

        # Для RAG творческий режим
        # не нужен.
        "temperature": 0.1,

        "max_tokens": (
            request.max_tokens
        ),

        "stream": False,

        "chat_template_kwargs": {
            "enable_thinking": False
        },
    }

    async with httpx.AsyncClient(
        timeout=180
    ) as client:

        response = await client.post(
            VLLM_URL,
            json=payload,
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=502,
            detail=(
                "Ошибка vLLM: "
                + response.text
            ),
        )

    llm_response = (
        response.json()
    )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    answer = (
        llm_response
        .get(
            "choices",
            [{}],
        )[0]
        .get(
            "message",
            {},
        )
        .get(
            "content",
            "",
        )
        .strip()
    )

    # Если модель сама придумала
    # section источников — удаляем.
    answer = (
        remove_generated_sources(
            answer
        )
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = get_source_documents(
        documents
    )

    if sources:

        sources_text = "\n".join(
            f"- {source}"
            for source in sources
        )

        answer += (
            "\n\n---\n"
            "**Использованные документы:**\n"
            f"{sources_text}"
        )

    # --------------------------------------------------------
    # OPENAI RESPONSE
    # --------------------------------------------------------

    llm_response[
        "choices"
    ][0][
        "message"
    ][
        "content"
    ] = answer

    llm_response["model"] = (
        RAG_MODEL_NAME
    )

    return llm_response