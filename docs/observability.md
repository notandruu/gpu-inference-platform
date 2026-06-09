# Observability

## Metrics sources

| Source | Endpoint | Content |
|---|---|---|
| FastAPI | `GET /metrics` | Request count, latency histograms, active requests |
| Triton | `:8003/metrics` | Inference count, queue duration, compute duration |
| DCGM Exporter | `:9400/metrics` | GPU utilization, memory used, temperature, bandwidth |

## Prometheus

Prometheus scrapes all three endpoints every 15 seconds (configured in `k8s/prometheus-config.yaml`).

Key metrics:

| Metric | Description |
|---|---|
| `api_requests_total` | Counter by endpoint and status code |
| `api_request_duration_seconds` | Histogram (p50/p95/p99) |
| `api_active_requests` | Gauge of in-flight requests |
| `triton_inference_duration_seconds` | Histogram of Triton gRPC call duration |
| `nv_inference_request_success` | Triton: total successful inferences |
| `nv_inference_queue_duration_us` | Triton: time spent in queue |
| `nv_inference_compute_infer_duration_us` | Triton: actual compute time |
| `DCGM_FI_DEV_GPU_UTIL` | GPU SM utilization % |
| `DCGM_FI_DEV_FB_USED` | GPU framebuffer memory used (MiB) |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | GPU memory bandwidth utilization % |

## Grafana dashboard

### Panels to configure

1. **API Request Rate** — `rate(api_requests_total[1m])`
2. **API p50 Latency** — `histogram_quantile(0.5, rate(api_request_duration_seconds_bucket[1m]))`
3. **API p95 Latency** — `histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[1m]))`
4. **API Error Rate** — `rate(api_requests_total{status=~"5.."}[1m]) / rate(api_requests_total[1m])`
5. **Triton Inference Rate** — `rate(nv_inference_request_success[1m])`
6. **Triton Queue Time** — `rate(nv_inference_queue_duration_us[1m])`
7. **Triton Compute Time** — `rate(nv_inference_compute_infer_duration_us[1m])`
8. **GPU Utilization** — `DCGM_FI_DEV_GPU_UTIL`
9. **GPU Memory Used** — `DCGM_FI_DEV_FB_USED`

### Local access

```bash
# Docker Compose
open http://localhost:3000   # admin / admin

# Kubernetes
kubectl port-forward svc/grafana 3000:3000 -n gpu-inference
open http://localhost:3000
```

Add Prometheus as a data source: `http://prometheus:9090` (Docker) or `http://prometheus:9090` (in-cluster).
