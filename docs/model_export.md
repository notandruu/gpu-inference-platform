# Model Export

## PyTorch → ONNX

```bash
make export-onnx
# or manually:
python models/export_onnx.py \
  --output model_repository/resnet50_onnx/1/model.onnx \
  --opset 17 \
  --device cpu
```

Options:
- `--opset` — ONNX opset version (17 = recent stable)
- `--device cpu|cuda` — run the PyTorch model on CPU or GPU during export
- `--no-dynamic-batch` — use a fixed batch size instead of a dynamic batch dimension

### Validate

```bash
make validate-model
# or manually:
python models/validate_model.py \
  --onnx model_repository/resnet50_onnx/1/model.onnx \
  --num-runs 5
```

Expected output:
```
run 1: max |PT - ONNX| = 0.000012
run 2: max |PT - ONNX| = 0.000009
...
[PASS] max difference across 5 runs: 0.000015
```

## ONNX → TensorRT

```bash
make build-tensorrt   # [GPU REQUIRED]
```

This calls `models/build_tensorrt.py` which tries the TensorRT Python API first, then falls back to `trtexec`.

Equivalent `trtexec` command:
```bash
trtexec \
  --onnx=model_repository/resnet50_onnx/1/model.onnx \
  --saveEngine=model_repository/resnet50_trt/1/model.plan \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:8x3x224x224 \
  --maxShapes=input:32x3x224x224 \
  --fp16
```

The engine is optimized for the GPU it is built on. Rebuild when moving to a different GPU generation.

Build logs are saved to `benchmarks/results/tensorrt_build.log`.
