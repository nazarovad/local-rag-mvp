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

Проверить документы после эмбеддинга `http://localhost:8001/debug/documents`

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
uvicorn rag-api.app.rag_api:app `
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
---
## Настройка в webUI
Панель администратора -> Настройки -> Подключения


<img width="1982" height="441" alt="image" src="https://github.com/user-attachments/assets/33c8aeb9-00f4-4640-9a8a-2bd687b37bb7" />
<img width="626" height="735" alt="image" src="https://github.com/user-attachments/assets/ff765249-9c5f-4dfa-a809-9bf978bf13a8" />





