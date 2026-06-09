from __future__ import annotations

import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferenceServerException

from app.config import settings
from app.metrics import TRITON_LATENCY


class TritonClient:
    def __init__(self) -> None:
        url = f"{settings.triton_host}:{settings.triton_grpc_port}"
        self._client = grpcclient.InferenceServerClient(url=url, verbose=False)

    def is_server_live(self) -> bool:
        try:
            return self._client.is_server_live()
        except Exception:
            return False

    def is_model_ready(self, model_name: str | None = None, version: str | None = None) -> bool:
        name = model_name or settings.model_name
        ver = version or settings.model_version
        try:
            return self._client.is_model_ready(name, ver)
        except Exception:
            return False

    def infer(
        self,
        input_data: np.ndarray,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> np.ndarray:
        name = model_name or settings.model_name
        version = model_version or settings.model_version

        infer_input = grpcclient.InferInput(
            settings.input_name, input_data.shape, "FP32"
        )
        infer_input.set_data_from_numpy(input_data)

        infer_output = grpcclient.InferRequestedOutput(settings.output_name)

        t0 = time.perf_counter()
        result = self._client.infer(
            model_name=name,
            model_version=version,
            inputs=[infer_input],
            outputs=[infer_output],
            timeout=settings.request_timeout_sec,
        )
        elapsed = time.perf_counter() - t0
        TRITON_LATENCY.labels(model=name).observe(elapsed)

        return result.as_numpy(settings.output_name)


# Module-level singleton; FastAPI app uses this.
_client: TritonClient | None = None


def get_client() -> TritonClient:
    global _client
    if _client is None:
        _client = TritonClient()
    return _client
