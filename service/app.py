from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .predictor import Predictor
from .schemas import InputItem, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель загружается ровно один раз при старте приложения.
    app.state.predictor = Predictor()
    yield
    app.state.predictor = None


app = FastAPI(
    title="Uzbek NER Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ok"}


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(items: list[InputItem]) -> PredictResponse:
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

    predictor: Predictor | None = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    records = [
        {
            "hash": item.hash,
            "text": item.text,
        }
        for item in items
    ]

    predictions = predictor.predict(records)
    return PredictResponse(data=predictions)
