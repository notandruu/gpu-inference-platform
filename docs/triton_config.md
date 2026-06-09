# Triton Model Repository & Configuration

## Repository layout

```
model_repository/
  resnet50_onnx/
    config.pbtxt        # ONNX Runtime backend config
    1/
      model.onnx        # exported from PyTorch (make export-onnx)
  resnet50_trt/
    config.pbtxt        # TensorRT Plan backend config
    1/
      model.plan        # built by TensorRT (make build-tensorrt)
```

The directory name is the model name. The `1/` subdirectory is model version 1.

## Dynamic batching

Both configs enable `dynamic_batching`:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 1000
}
```

- Triton queues incoming requests and waits up to 1 ms to coalesce them into a batch of 4, 8, or 16 before dispatching a single GPU kernel call.
- Under light load (single client), Triton dispatches immediately (batch=1).
- Under heavy concurrent load, batching improves GPU utilization significantly.

## max_batch_size

Set to 32. Triton will reject requests with batch > 32. The TensorRT engine is built with `--maxShapes=input:32x3x224x224` to match.

## GPU instance group

```protobuf
instance_group [
  { count: 1, kind: KIND_GPU }
]
```

Increase `count` for multiple GPU instances on a single GPU (limited by VRAM) or change `kind: KIND_CPU` for CPU-only testing.

## Switching models

FastAPI uses the `MODEL_NAME` environment variable:

```bash
MODEL_NAME=resnet50_onnx make run-api   # use ONNX backend
MODEL_NAME=resnet50_trt  make run-api   # use TensorRT backend
```

## Checking model status

```bash
curl http://localhost:8002/v2/models/resnet50_trt/ready
curl http://localhost:8002/v2/models/resnet50_onnx/ready
curl http://localhost:8002/v2/health/ready
```
