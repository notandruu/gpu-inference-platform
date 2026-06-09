# Nsight Compute Profiling

Nsight Compute provides per-kernel analysis: memory throughput, roofline model, warp efficiency, occupancy, and instruction mix.

## Install

```bash
# Ships with CUDA Toolkit 12+
ncu --version
```

## Profile the CUDA preprocessing kernel

```bash
# Profile all kernels launched during preprocessing
ncu \
  --output profiling/reports/preprocess_kernel \
  --force-overwrite \
  --set full \
  python -c "
import numpy as np
import cuda_preprocess as cp
imgs = np.random.randint(0, 256, (16, 224, 224, 3), dtype='uint8')
out = cp.preprocess(imgs)
print(out.shape)
"

# Open report in Nsight Compute GUI
ncu-ui profiling/reports/preprocess_kernel.ncu-rep
```

## Profile a specific kernel by name

```bash
ncu \
  --kernel-name nhwc_to_nchw_normalize_kernel \
  --output profiling/reports/preprocess_kernel_targeted \
  --set roofline,memory,launch,scheduler \
  python -c "
import numpy as np
import cuda_preprocess as cp
imgs = np.random.randint(0, 256, (32, 224, 224, 3), dtype='uint8')
cp.preprocess(imgs)
"
```

## CLI summary without GUI

```bash
ncu \
  --kernel-name nhwc_to_nchw_normalize_kernel \
  --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    gpu__time_duration.sum \
  python -c "
import numpy as np
import cuda_preprocess as cp
imgs = np.random.randint(0, 256, (32, 224, 224, 3), dtype='uint8')
cp.preprocess(imgs)
" 2>&1 | tee profiling/reports/preprocess_metrics.txt
```

## Key metrics to analyze

| Metric | Description | Target |
|---|---|---|
| `sm__throughput` | SM compute utilization | >80% for compute-bound |
| `dram__throughput` | Memory bandwidth utilization | >80% for memory-bound |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld` | Global load bytes | Minimize with coalesced access |
| `sm__warps_active` | Active warps / peak | Higher = better occupancy |
| Roofline position | Below compute or memory ceiling | Optimize whichever ceiling you hit |

## Interpreting results

The `nhwc_to_nchw_normalize_kernel` is a simple per-pixel transform; it is likely **memory-bandwidth-bound**:

- If `dram__throughput` is near 100%, the kernel is at the memory roofline — pinned memory and larger batches won't help much.
- If `sm__throughput` is low with `dram__throughput` also low, the kernel has access pattern issues (non-coalesced reads from NHWC layout).

**Optimization to try:** restructure the NHWC input to improve coalesced global loads, or use shared memory tiling.

## Compare batch sizes

```bash
for BS in 1 4 8 16 32; do
  echo "=== batch=$BS ===" >> profiling/reports/batch_comparison.txt
  ncu \
    --kernel-name nhwc_to_nchw_normalize_kernel \
    --metrics gpu__time_duration.sum \
    python -c "
import numpy as np, cuda_preprocess as cp
imgs = np.random.randint(0, 256, ($BS, 224, 224, 3), dtype='uint8')
cp.preprocess(imgs)
" 2>&1 >> profiling/reports/batch_comparison.txt
done
```
