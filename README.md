# GPU Inference Platform

Production-style GPU inference platform using PyTorch, ONNX, TensorRT, NVIDIA Triton, FastAPI, Kubernetes, Prometheus/Grafana, and Nsight profiling.

## Architecture

```
                       ┌────────────────────────┐
                       │   Benchmark Client      │
                       │   asyncio / httpx       │
                       └───────────┬────────────┘
                                   │ HTTP
                                   ▼
                       ┌────────────────────────┐
                       │   FastAPI Ingress       │
                       │   /predict  /batch      │
                       │   /health   /ready      │
                       │   /metrics (Prometheus) │
                       └───────────┬────────────┘
                                   │ gRPC
                                   ▼
                       ┌────────────────────────┐
                       │  NVIDIA Triton Server   │
                       │  Dynamic batching       │
                       │  Model versioning       │
                       └────────────┬───────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              ▼                                            ▼
   ┌──────────────────────┐                  ┌────────────────────────┐
   │  resnet50_onnx        │                  │  resnet50_trt          │
   │  ONNX Runtime backend │                  │  TensorRT Plan (FP16)  │
   └──────────────────────┘                  └────────────────────────┘

Observability: FastAPI /metrics + Triton /metrics + DCGM Exporter → Prometheus → Grafana
```

## Tech stack

| Layer | Technologies |
|---|---|
| ML / Inference | PyTorch, ONNX, ONNX Runtime, TensorRT, NVIDIA Triton |
| API | FastAPI, Uvicorn, Pydantic, tritonclient[grpc] |
| GPU extension | C++17, CUDA, pybind11 |
| Containers | Docker, Docker Compose, NVIDIA Container Toolkit |
| Orchestration | Kubernetes, NVIDIA device plugin, HPA |
| Observability | Prometheus, Grafana, NVIDIA DCGM Exporter |
| Profiling | Nsight Systems, Nsight Compute, perf_analyzer |

## Quick start (local, CPU only)

```bash
# 1. Install deps
make setup
source .venv/bin/activate

# 2. Export ONNX model (~100 MB, downloads pretrained weights once)
make export-onnx

# 3. Validate ONNX vs PyTorch numerical agreement
make validate-model

# 4. Start FastAPI (talks to Triton; set MODEL_NAME=resnet50_onnx for local ONNX RT testing)
MODEL_NAME=resnet50_onnx make run-api

# 5. Test a prediction (in another terminal)
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@benchmarks/sample_payloads/dog.jpg"
```

## GPU workflow

```bash
# Requires: NVIDIA GPU + CUDA + Docker + NVIDIA Container Toolkit

# Build TensorRT engine
make build-tensorrt

# Start Triton + API + observability
make docker-up

# Run benchmark
make benchmark

# Check results
cat benchmarks/results/backend_comparison.md
```

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/ready` | GET | Readiness + Triton connectivity |
| `/metrics` | GET | Prometheus metrics |
| `/v1/predict` | POST | Single-image inference |
| `/v1/batch_predict` | POST | Multi-image batch inference |
| `/v1/benchmark_once` | POST | Single timed request |

### Example

```bash
# Health
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/v1/predict \
  -F "file=@path/to/image.jpg"

# Response
{
  "model": "resnet50_trt",
  "version": "1",
  "predictions": [
    {"class_id": 207, "label": "golden retriever", "confidence": 0.923},
    ...
  ],
  "latency_ms": 4.2
}
```

## Model export

See [docs/model_export.md](docs/model_export.md) for full details.

```bash
make export-onnx       # PyTorch → ONNX
make validate-model    # compare PyTorch vs ONNX Runtime numerically
make build-tensorrt    # ONNX → TensorRT FP16 engine  [GPU REQUIRED]
```

## Triton configuration

See [docs/triton_config.md](docs/triton_config.md).

Both models (`resnet50_onnx`, `resnet50_trt`) use dynamic batching with preferred batch sizes `[4, 8, 16]` and a 1 ms queue delay. Switch backends by setting `MODEL_NAME`.

## Benchmark

```bash
make benchmark

# Or directly:
python benchmarks/run_backend_comparison.py \
  --concurrency 1 4 8 16 32 \
  --batch-size 1 4 8 \
  --duration-sec 60 \
  --output benchmarks/results/backend_comparison.json

python benchmarks/analyze_results.py
```

Results are written to `benchmarks/results/` (excluded from git — run locally to generate).

## Docker Compose

```bash
make docker-up     # starts triton, api, prometheus, grafana
make docker-down

# Endpoints after up:
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
```

Note: Triton requires `--gpus all` / NVIDIA Container Toolkit. On a CPU-only machine, comment out the `deploy.resources` block in `docker-compose.yml` and use `MODEL_NAME=resnet50_onnx` with a CPU instance group.

## Kubernetes

See [docs/kubernetes_deployment.md](docs/kubernetes_deployment.md).

```bash
make k8s-deploy

kubectl get pods -n gpu-inference
kubectl port-forward svc/gpu-inference-api 8000:8000 -n gpu-inference
kubectl port-forward svc/grafana 3000:3000 -n gpu-inference
```

Triton deployment requests `nvidia.com/gpu: 1`. The FastAPI tier runs without GPU and scales horizontally via HPA.

## Observability

See [docs/observability.md](docs/observability.md).

Grafana dashboard panels: API request rate, p50/p95 latency, error rate, Triton queue/compute time, GPU utilization, GPU memory used.

## C++/CUDA preprocessing extension

```bash
# [GPU REQUIRED]
make build-cuda

# Test
pytest cuda_preprocess/tests/

# Benchmark preprocessing: CUDA vs NumPy/PyTorch baseline
python benchmarks/run_backend_comparison.py
```

The CUDA kernel (`cuda_preprocess/preprocess_kernel.cu`) performs batched NHWC uint8 → NCHW float32 conversion with ImageNet normalization. Profiled with Nsight Compute to analyze memory bandwidth utilization.

See [profiling/nsight_compute_commands.md](profiling/nsight_compute_commands.md).

## Profiling

```bash
# Nsight Systems (end-to-end timeline)
# See profiling/nsight_systems_commands.md

# Nsight Compute (per-kernel analysis)
# See profiling/nsight_compute_commands.md
```

Profiling reports go in `profiling/reports/` (excluded from git).

## Tests

```bash
make test         # runs all non-GPU, non-Triton tests
pytest -v         # verbose output
```

## Benchmark results

CPU baselines (Apple M-series, batch=1–8, concurrency=1). Run `make benchmark` to regenerate.

| Backend | Batch | Concurrency | p50 ms | p95 ms | p99 ms | Tput req/s |
|---|---:|---:|---:|---:|---:|---:|
| pytorch_local | 1 | 1 | 18.0 | 18.4 | 18.8 | 51.6 |
| pytorch_local | 4 | 1 | 48.5 | 50.7 | 56.3 | 19.3 |
| pytorch_local | 8 | 1 | 77.7 | 80.6 | 83.3 | 11.3 |
| onnxruntime_local | 1 | 1 | 16.2 | 18.9 | 31.6 | 60.0 |
| onnxruntime_local | 4 | 1 | 77.1 | 81.5 | 97.3 | 12.9 |
| onnxruntime_local | 8 | 1 | 119.1 | 130.0 | 168.9 | 8.4 |
| triton_trt (GPU) | 1 | 1–32 | *run on GPU* | — | — | — |
| triton_trt + dynbatch (GPU) | 8 | 16 | *run on GPU* | — | — | — |

> Triton/TensorRT rows require a GPU node. Run `make benchmark-full` with a live Triton server to populate them.

Preprocessing benchmark (CPU, 50 runs each):

| Backend | Batch=1 p50 ms | Batch=8 p50 ms | Batch=32 p50 ms |
|---|---:|---:|---:|
| NumPy | 0.57 | 4.77 | 19.55 |
| PyTorch (CPU) | 0.23 | 0.44 | 1.28 |
| CUDA kernel | *GPU required* | — | — |

## Resume bullets

See [docs/resume_bullets.md](docs/resume_bullets.md) for NVIDIA AI infrastructure, NVIDIA systems software, and generic SWE versions.

## Project structure

```
gpu-inference-platform/
  app/                    FastAPI service + Triton client + preprocessing
  models/                 Export, build, and validate scripts
  model_repository/       Triton model configs and artifacts
  cuda_preprocess/        C++/CUDA extension + pybind11 bindings
  benchmarks/             Benchmark harness and results
  docker/                 Dockerfiles
  k8s/                    Kubernetes manifests
  profiling/              Nsight profiling commands and reports
  tests/                  Unit tests (CPU, no GPU required)
  scripts/                Shell helpers (setup, build, run, deploy)
  docs/                   Architecture, export, config, k8s, observability docs
```

## Checklist

- [x] `make export-onnx`
- [x] `make validate-model`
- [ ] `make build-tensorrt` — GPU required
- [ ] `make run-triton` — GPU required
- [x] `make run-api` — CPU works (Triton mocked)
- [x] `curl /health`
- [ ] `curl /ready` — requires live Triton
- [ ] `/v1/predict` — requires live Triton
- [x] `make benchmark` — local PyTorch/ONNX baselines work; Triton backends require GPU
- [x] `docker-compose.yml` — GPU deployment configured
- [x] Kubernetes manifests with `nvidia.com/gpu: 1`
- [x] Prometheus/Grafana/DCGM configuration
- [x] CUDA preprocessing extension (build requires GPU)
- [x] Nsight profiling docs
