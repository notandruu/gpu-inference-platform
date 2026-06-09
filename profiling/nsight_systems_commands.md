# Nsight Systems Profiling

Nsight Systems captures the full timeline: CPU threads, CUDA kernel launches, memory copies, and NVTX annotations.

## Install

```bash
# On DGX/cloud GPU node — Nsight Systems ships with CUDA Toolkit 12+
nsys --version

# Or download the standalone installer from NVIDIA Developer
# https://developer.nvidia.com/nsight-systems
```

## Profile Triton + FastAPI end-to-end

Run the FastAPI server under `nsys`. NVTX annotations (if added) will appear in the trace.

```bash
# Start Triton in one terminal
bash scripts/run_triton_local.sh

# Profile the FastAPI API server process
nsys profile \
  --output profiling/reports/api_profile \
  --trace cuda,nvtx,osrt,cudnn,cublas \
  --force-overwrite true \
  uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal, run a benchmark to generate load
python benchmarks/run_backend_comparison.py \
  --duration-sec 30 \
  --concurrency 4 \
  --output benchmarks/results/profiling_run.json

# Open report in Nsight Systems GUI
nsys-ui profiling/reports/api_profile.nsys-rep
```

## Profile a single Triton inference request

```bash
# Use tritonclient to send one request; wrap the client script
nsys profile \
  --output profiling/reports/triton_single_request \
  --trace cuda,nvtx \
  python -c "
import numpy as np, tritonclient.grpc as grpc
c = grpc.InferenceServerClient('localhost:8001')
inp = grpc.InferInput('input', [1, 3, 224, 224], 'FP32')
inp.set_data_from_numpy(np.random.randn(1, 3, 224, 224).astype('float32'))
out = grpc.InferRequestedOutput('output')
r = c.infer('resnet50_trt', [inp], outputs=[out])
print(r.as_numpy('output').shape)
"
```

## Profile CUDA preprocessing kernel

```bash
nsys profile \
  --output profiling/reports/cuda_preprocess \
  --trace cuda,nvtx \
  python -c "
import numpy as np
import cuda_preprocess as cp
imgs = np.random.randint(0, 256, (16, 224, 224, 3), dtype='uint8')
for _ in range(100):
    out = cp.preprocess(imgs)
print('done', out.shape)
"
```

## Key metrics to look for in timeline

| Signal | What to look for |
|---|---|
| CPU→GPU memory copy duration | Should shrink with pinned memory |
| CUDA kernel time | Compare normalized vs unnormalized preprocessing |
| Triton queue time | Visible gap between request receipt and kernel start |
| Dynamic batching coalescing | Multiple requests coalesced into one kernel call |
| CPU idle time | FastAPI async awaiting Triton gRPC response |

## Tips

- Add `torch.cuda.nvtx.range_push("label")` / `range_pop()` in PyTorch code to annotate the timeline
- Use `--capture-range=cudaProfilerApi` to only profile specific code regions
- `--stats=true` generates a summary report without requiring the GUI
