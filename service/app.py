from __future__ import annotations

import hashlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .predictor import Predictor
from .schemas import InputItem, PredictRequest


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_predictor(app: FastAPI) -> Predictor:
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready",
        )
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель, токенизатор и train-лексиконы загружаются один раз
    # до того, как сервис начинает принимать запросы.
    app.state.predictor = Predictor()
    yield
    app.state.predictor = None


app = FastAPI(
    title="Uzbek NER API",
    version="2.1.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------
# Обязательный endpoint по контракту кейса.
# После завершения startup модель уже готова, поэтому здесь
# нет инференса и других тяжёлых операций.
# ------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# Дополнительный endpoint для локальной диагностики.
@app.get("/health")
def health() -> dict:
    predictor = _get_predictor(app)
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(predictor.device),
    }


@app.get("/model-info")
def model_info() -> dict:
    predictor = _get_predictor(app)
    return predictor.info()


# ------------------------------------------------------------
# Обязательный endpoint по контракту кейса.
# Вход: непустой JSON-массив [{hash, text}, ...]
# Выход: {"data": [{hash, entities: [{label,start,end}, ...]}, ...]}
# ------------------------------------------------------------
@app.post("/api/v1/predict")
def predict_batch(items: list[InputItem]) -> dict:
    if not items:
        raise HTTPException(
            status_code=422,
            detail="Request body must be a non-empty JSON array",
        )

    hashes = [item.hash for item in items]
    if len(hashes) != len(set(hashes)):
        raise HTTPException(
            status_code=422,
            detail="hash must be unique within request",
        )

    predictor = _get_predictor(app)

    records = [
        {
            "hash": item.hash,
            "text": item.text,
        }
        for item in items
    ]

    try:
        predictions, _ = predictor.predict(records)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    data = []

    for source_record, prediction in zip(records, predictions):
        text = source_record["text"]
        entities = []
        seen = set()

        for entity in prediction["entities"]:
            label = str(entity["label"])
            start = int(entity["start"])
            end = int(entity["end"])

            if label not in {"ORG", "NAME", "GEO"}:
                continue

            if not (0 <= start < end <= len(text)):
                continue

            key = (label, start, end)
            if key in seen:
                continue
            seen.add(key)

            entities.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                }
            )

        entities.sort(
            key=lambda entity: (
                entity["start"],
                entity["end"],
                entity["label"],
            )
        )

        data.append(
            {
                "hash": source_record["hash"],
                "entities": entities,
            }
        )

    return {"data": data}


# ------------------------------------------------------------
# Дополнительный endpoint только для Streamlit-демо.
# Он не является частью обязательного контракта кейса.
# ------------------------------------------------------------
@app.post("/predict")
def predict_demo(request: PredictRequest) -> dict:
    text = request.text

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is empty",
        )

    predictor = _get_predictor(app)

    record = {
        "hash": _hash_text(text),
        "text": text,
    }

    started = time.perf_counter()

    try:
        predictions, stats = predictor.predict([record])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    latency_ms = (time.perf_counter() - started) * 1000.0

    entities = []
    for entity in predictions[0]["entities"]:
        start = int(entity["start"])
        end = int(entity["end"])

        entities.append(
            {
                "text": text[start:end],
                "label": str(entity["label"]),
                "start": start,
                "end": end,
                "confidence": entity.get("confidence"),
                "source": entity.get("source", "model"),
            }
        )

    return {
        "text": text,
        "entities": entities,
        "latency_ms": latency_ms,
        "postprocess_stats": {
            key: int(value)
            for key, value in stats.items()
        },
    }
