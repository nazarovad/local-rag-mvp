from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests


MODEL_PATH = "models/bge-m3"
COLLECTION = "documents"

VLLM_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "qwen3-8b"


def main():
    model = SentenceTransformer(
        MODEL_PATH,
        local_files_only=True
    )

    qdrant = QdrantClient(
        url="http://localhost:6333"
    )

    question = input("Вопрос: ").strip()

    vector = model.encode(
        question,
        normalize_embeddings=True
    )

    result = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector.tolist(),
        limit=5,
        with_payload=True
    )

    points = result.points

    print()
    print("Найденные фрагменты:")
    print("=" * 80)

    contexts = []

    for i, point in enumerate(points, start=1):
        text = point.payload["text"]

        contexts.append(text)

        print(f"\n--- Фрагмент {i}, score={point.score:.4f} ---")
        print(text[:1000])

    context = "\n\n---\n\n".join(contexts)

    system_prompt = """
Ты помощник по корпоративным документам.

Отвечай на русском языке.
Используй только информацию из предоставленного контекста.
Не придумывай факты, суммы, даты или условия.

Если в контексте нет ответа, скажи:
"В предоставленном документе информация не найдена."

Если ответ связан с таблицей, внимательно учитывай
заголовки колонок и соответствующие значения.
"""

    user_prompt = f"""
КОНТЕКСТ:

{context}

ВОПРОС:

{question}
"""

    response = requests.post(
        VLLM_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 800,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        timeout=120
    )

    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]

    print()
    print("=" * 80)
    print("ОТВЕТ:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()