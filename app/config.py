import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    triton_host: str = os.getenv("TRITON_HOST", "localhost")
    triton_grpc_port: int = int(os.getenv("TRITON_GRPC_PORT", "8001"))
    triton_http_port: int = int(os.getenv("TRITON_HTTP_PORT", "8000"))

    # Default to TensorRT model; fall back to ONNX when plan is unavailable
    model_name: str = os.getenv("MODEL_NAME", "resnet50_trt")
    model_version: str = os.getenv("MODEL_VERSION", "1")

    input_name: str = "input"
    output_name: str = "output"
    input_shape: tuple[int, int, int] = (3, 224, 224)  # C H W

    top_k: int = 5
    request_timeout_sec: float = 10.0

    metrics_port: int = int(os.getenv("METRICS_PORT", "8000"))

    class Config:
        env_file = ".env"


settings = Settings()
