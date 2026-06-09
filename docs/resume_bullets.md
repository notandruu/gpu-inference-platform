# Resume Bullets

## NVIDIA AI Infrastructure Version

```latex
\resumeProjectHeading
  {\textbf{GPU Inference Platform} $|$ \emph{Python, C++, PyTorch, CUDA, ONNX, TensorRT, Triton, Kubernetes}}{\small ML Systems Project}
  \resumeItemListStart
    \resumeItem{Built a \textbf{GPU-accelerated inference platform} by exporting PyTorch models to ONNX and TensorRT engines, then serving optimized model versions through \textbf{NVIDIA Triton Inference Server} with a FastAPI ingress layer}
    \resumeItem{Implemented \textbf{dynamic batching}, async request handling, model versioning, health checks, and benchmark harnesses to measure p50/p95 latency, throughput, cold-start behavior, and GPU utilization}
    \resumeItem{Containerized inference services with Docker and deployed on \textbf{Kubernetes} using GPU node scheduling (\texttt{nvidia.com/gpu}), Prometheus/Grafana dashboards, and DCGM exporter for GPU memory and utilization monitoring}
    \resumeItem{Built and profiled a \textbf{C++/CUDA} preprocessing extension for batched image normalization and NHWC→NCHW layout conversion, using Nsight Systems/Compute to identify CPU-GPU transfer and kernel bottlenecks}
  \resumeItemListEnd
```

## NVIDIA Systems Software Version

```latex
\resumeProjectHeading
  {\textbf{GPU Inference Platform} $|$ \emph{C++, CUDA, TensorRT, Python, Kubernetes, Prometheus}}{\small ML Systems Project}
  \resumeItemListStart
    \resumeItem{Designed end-to-end GPU inference pipeline: PyTorch → ONNX → \textbf{TensorRT FP16 engine} → NVIDIA Triton with dynamic batching, achieving measurable latency reduction vs.\ ONNX Runtime baseline}
    \resumeItem{Wrote \textbf{C++/CUDA kernel} for batched image normalization and NHWC→NCHW layout conversion; profiled with \textbf{Nsight Compute} to analyze memory bandwidth utilization and warp efficiency}
    \resumeItem{Deployed serving stack on \textbf{Kubernetes} with GPU device plugin scheduling, DCGM exporter DaemonSet, Prometheus scraping, and HPA for the CPU-bound FastAPI ingress tier}
  \resumeItemListEnd
```

## Generic SWE Version

```latex
\resumeProjectHeading
  {\textbf{GPU Inference Platform} $|$ \emph{Python, Docker, Kubernetes, FastAPI, Prometheus}}{\small ML Systems Project}
  \resumeItemListStart
    \resumeItem{Built a production-style model serving platform with FastAPI, NVIDIA Triton, Docker Compose, and Kubernetes; exposed Prometheus metrics and Grafana dashboards for latency and throughput observability}
    \resumeItem{Automated model optimization pipeline (PyTorch → ONNX → TensorRT) and benchmark harness measuring p50/p95/p99 latency across concurrency levels and batch sizes}
  \resumeItemListEnd
```
