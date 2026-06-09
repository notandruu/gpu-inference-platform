#!/usr/bin/env bash
# Full build pipeline: ONNX export → TensorRT engine.
# GPU is required for TensorRT.
set -euo pipefail

echo "==> Exporting ONNX model..."
python models/export_onnx.py

echo "==> Validating ONNX model..."
python models/validate_model.py

echo "==> Building TensorRT engine [GPU REQUIRED]..."
bash scripts/build_tensorrt.sh

echo ""
echo "Build complete. Start serving:"
echo "  make docker-up"
echo "  make run-triton  # local GPU"
