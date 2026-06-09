from app.triton_client import get_client
from app.config import settings


def liveness() -> dict:
    return {"status": "ok"}


def readiness() -> dict:
    client = get_client()
    live = client.is_server_live()
    ready = client.is_model_ready() if live else False
    return {
        "status": "ok" if (live and ready) else "degraded",
        "triton_connected": live,
        "model_ready": ready,
    }
