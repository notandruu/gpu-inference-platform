from pydantic import BaseModel, Field


class Prediction(BaseModel):
    class_id: int
    label: str
    confidence: float


class PredictResponse(BaseModel):
    model: str
    version: str
    predictions: list[Prediction]
    latency_ms: float


class BatchPredictResponse(BaseModel):
    model: str
    version: str
    results: list[list[Prediction]]
    latency_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    status: str
    triton_connected: bool
    model_ready: bool


class BenchmarkOnceResponse(BaseModel):
    model: str
    latency_ms: float
    predictions: list[Prediction]


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
