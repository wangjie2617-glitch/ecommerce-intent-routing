"""FastAPI entrypoint for the intent classification and routing service."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.config import MAX_BATCH_SIZE
from app.model_service import ModelNotReadyError, model_service
from app.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    PredictionResponse,
    PredictRequest,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_service.load()
    except ModelNotReadyError:
        pass
    yield


app = FastAPI(
    title="电商客户诉求识别与智能工单路由",
    version="1.0.0",
    description="多标签文本分类、风险识别与可解释工单路由工程项目。",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if model_service.is_ready else "model_not_ready",
        "model_type": model_service.model_type,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> dict[str, object]:
    try:
        return model_service.predict([request.text])[0]
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictRequest) -> dict[str, object]:
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"批量请求最多支持 {MAX_BATCH_SIZE} 条")
    started = time.perf_counter()
    try:
        items = model_service.predict(request.texts)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "items": items,
        "count": len(items),
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
