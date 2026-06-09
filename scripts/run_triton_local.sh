#!/usr/bin/env bash
# [GPU REQUIRED] Run Triton Inference Server locally via Docker.
set -euo pipefail

MODEL_REPO="$(pwd)/model_repository"

docker run --rm --gpus=all \
  -p 8001:8001 -p 8002:8002 -p 8003:8003 \
  -v "${MODEL_REPO}:/models:ro" \
  nvcr.io/nvidia/tritonserver:24.04-py3 \
  tritonserver \
    --model-repository=/models \
    --allow-metrics=true \
    --log-verbose=0

echo ""
echo "Triton gRPC:    localhost:8001"
echo "Triton HTTP:    localhost:8002"
echo "Triton metrics: localhost:8003/metrics"
