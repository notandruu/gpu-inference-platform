from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

registry = CollectorRegistry(auto_describe=True)

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry,
)

TRITON_LATENCY = Histogram(
    "triton_inference_duration_seconds",
    "Triton inference latency",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

BATCH_SIZE = Histogram(
    "api_batch_size",
    "Inference batch sizes",
    buckets=[1, 2, 4, 8, 16, 32, 64],
    registry=registry,
)

ACTIVE_REQUESTS = Gauge(
    "api_active_requests",
    "Currently active requests",
    registry=registry,
)


def get_metrics() -> tuple[bytes, str]:
    return generate_latest(registry), CONTENT_TYPE_LATEST
