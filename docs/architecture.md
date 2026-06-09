# Architecture

```
                       ┌────────────────────────┐
                       │   Benchmark Client      │
                       │   asyncio/httpx/Locust  │
                       └───────────┬────────────┘
                                   │ HTTP/REST
                                   ▼
                       ┌────────────────────────┐
                       │   FastAPI Ingress       │
                       │   /predict              │
                       │   /batch_predict        │
                       │   /health  /ready       │
                       │   /metrics (Prometheus) │
                       └───────────┬────────────┘
                                   │ gRPC (tritonclient)
                                   ▼
                       ┌────────────────────────┐
                       │  NVIDIA Triton Server   │
                       │  Dynamic batching       │
                       │  Model versioning       │
                       │  ONNX + TensorRT backend│
                       └────────────┬───────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              ▼                                            ▼
   ┌──────────────────────┐                  ┌────────────────────────┐
   │  resnet50_onnx        │                  │  resnet50_trt          │
   │  ONNX Runtime backend │                  │  TensorRT Plan backend │
   │  model.onnx           │                  │  model.plan (FP16)     │
   └──────────────────────┘                  └────────────────────────┘

Observability:
FastAPI /metrics ──────┐
Triton /metrics ───────┼──► Prometheus ──► Grafana
DCGM Exporter /metrics ┘

Profiling:
Nsight Systems / Nsight Compute / nvidia-smi / Triton perf_analyzer
```

## Request flow

1. Client sends HTTP POST with image to FastAPI `/v1/predict`
2. FastAPI decodes and preprocesses the image (Python NumPy or CUDA extension)
3. FastAPI calls Triton over gRPC via `tritonclient`
4. Triton queues the request, applies dynamic batching with concurrent requests
5. Triton runs inference on the selected backend (ONNX or TensorRT)
6. Triton returns logits to FastAPI
7. FastAPI applies softmax and returns top-5 predictions as JSON

## Kubernetes layout

```
Namespace: gpu-inference
  Deployment: triton           (1 replica, nvidia.com/gpu: 1)
  Deployment: gpu-inference-api (2+ replicas, HPA CPU target 70%)
  DaemonSet:  dcgm-exporter    (GPU nodes only)
  Deployment: prometheus
  Deployment: grafana
```
