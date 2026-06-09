#!/usr/bin/env bash
# [GPU REQUIRED] Build TensorRT engine from ONNX model.
set -euo pipefail

ONNX_PATH="${ONNX_PATH:-model_repository/resnet50_onnx/1/model.onnx}"
ENGINE_PATH="${ENGINE_PATH:-model_repository/resnet50_trt/1/model.plan}"
LOG_PATH="${LOG_PATH:-benchmarks/results/tensorrt_build.log}"

if [ ! -f "$ONNX_PATH" ]; then
  echo "ONNX model not found: $ONNX_PATH"
  echo "Run: make export-onnx"
  exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"
mkdir -p "$(dirname "$LOG_PATH")"

# Try Python API first; fall back to trtexec
python models/build_tensorrt.py \
  --onnx "$ONNX_PATH" \
  --output "$ENGINE_PATH" \
  --log "$LOG_PATH" \
  --fp16

echo "TensorRT engine ready: $ENGINE_PATH"
