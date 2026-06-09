import time
import pathlib
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    HealthResponse,
    ReadyResponse,
    PredictResponse,
    BatchPredictResponse,
    BenchmarkOnceResponse,
    Prediction,
)
from app.preprocessing import preprocess_image, preprocess_batch
from app.triton_client import get_client
from app.health import liveness, readiness
from app.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    ACTIVE_REQUESTS,
    BATCH_SIZE,
    get_metrics,
)

_LABELS_PATH = pathlib.Path(__file__).parent.parent / "models" / "labels.txt"
_LABELS: list[str] = []


def _load_labels() -> list[str]:
    if _LABELS_PATH.exists():
        return [ln.strip() for ln in _LABELS_PATH.read_text().splitlines()]
    return [str(i) for i in range(1000)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _LABELS
    _LABELS = _load_labels()
    yield


app = FastAPI(
    title="GPU Inference Platform",
    version="0.1.0",
    description="FastAPI ingress for NVIDIA Triton Inference Server",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _top_k_predictions(logits: np.ndarray, k: int = 5) -> list[Prediction]:
    probs = _softmax(logits.flatten())
    indices = np.argsort(probs)[::-1][:k]
    return [
        Prediction(
            class_id=int(idx),
            label=_LABELS[idx] if idx < len(_LABELS) else str(idx),
            confidence=float(probs[idx]),
        )
        for idx in indices
    ]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@app.get("/health", response_model=HealthResponse)
def health():
    data = liveness()
    REQUEST_COUNT.labels(endpoint="/health", status="200").inc()
    return data


@app.get("/ready", response_model=ReadyResponse)
def ready():
    data = readiness()
    status = "200" if data["status"] == "ok" else "503"
    REQUEST_COUNT.labels(endpoint="/ready", status=status).inc()
    if data["status"] != "ok":
        raise HTTPException(status_code=503, detail=data)
    return data


@app.get("/metrics")
def metrics():
    content, content_type = get_metrics()
    return Response(content=content, media_type=content_type)


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)  # (1, 3, 224, 224)

        client = get_client()
        logits = client.infer(tensor)  # (1, 1000)

        predictions = _top_k_predictions(logits[0], k=settings.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        REQUEST_COUNT.labels(endpoint="/v1/predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/v1/predict").observe(latency_ms / 1000)

        return PredictResponse(
            model=settings.model_name,
            version=settings.model_version,
            predictions=predictions,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/v1/predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/v1/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(files: list[UploadFile] = File(...)):
    ACTIVE_REQUESTS.inc()
    t0 = time.perf_counter()
    try:
        images_bytes = [await f.read() for f in files]
        batch = preprocess_batch(images_bytes)  # (N, 3, 224, 224)

        BATCH_SIZE.observe(len(images_bytes))
        client = get_client()
        logits = client.infer(batch)  # (N, 1000)

        results = [_top_k_predictions(logits[i], k=settings.top_k) for i in range(len(files))]
        latency_ms = (time.perf_counter() - t0) * 1000

        REQUEST_COUNT.labels(endpoint="/v1/batch_predict", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/v1/batch_predict").observe(latency_ms / 1000)

        return BatchPredictResponse(
            model=settings.model_name,
            version=settings.model_version,
            results=results,
            latency_ms=latency_ms,
            batch_size=len(files),
        )
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/v1/batch_predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/v1/benchmark_once", response_model=BenchmarkOnceResponse)
async def benchmark_once(file: UploadFile = File(...)):
    """Single timed request useful for sanity-checking latency interactively."""
    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes)

    client = get_client()
    t0 = time.perf_counter()
    logits = client.infer(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000

    return BenchmarkOnceResponse(
        model=settings.model_name,
        latency_ms=latency_ms,
        predictions=_top_k_predictions(logits[0], k=settings.top_k),
    )
