FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 1. Ставим зависимости отдельно, чтобы Docker мог кэшировать слой.
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 2. Во время BUILD скачиваем модель и токенизатор и сохраняем
#    их обычными локальными файлами внутри Docker image.
#    Во время docker run интернет уже не требуется.
RUN python - <<'PY'
from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer

model_dir = Path("/app/model/uzbek-ner")
tokenizer_dir = Path("/app/model/xlm-roberta-large-tokenizer")

model_dir.mkdir(parents=True, exist_ok=True)
tokenizer_dir.mkdir(parents=True, exist_ok=True)

print("[build] Downloading NER model...")
model = AutoModelForTokenClassification.from_pretrained(
    "dashakoryakovskaya/uzbek-ner",
)
model.save_pretrained(
    model_dir,
    safe_serialization=True,
)

print("[build] Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/xlm-roberta-large",
    use_fast=True,
)
tokenizer.save_pretrained(tokenizer_dir)

print("[build] Model resources saved inside image.")
PY

# 3. Кладём только то, что требуется для обязательного HTTP-сервиса.
COPY service /app/service
COPY data/train.jsonl /app/data/train.jsonl

# 4. На runtime принудительно используем только локальные ресурсы.
ENV NER_MODEL_NAME=/app/model/uzbek-ner \
    NER_TOKENIZER_NAME=/app/model/xlm-roberta-large-tokenizer \
    NER_TRAIN_PATH=/app/data/train.jsonl \
    NER_DEVICE=cpu \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# По контракту сервис должен слушать 0.0.0.0:8000.
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
