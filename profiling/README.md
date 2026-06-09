# Profiling

This directory contains Nsight profiling documentation and saved reports.

## Tools

| Tool | Purpose |
|---|---|
| Nsight Systems (`nsys`) | End-to-end timeline: CPU, GPU, CUDA kernels, API calls |
| Nsight Compute (`ncu`) | Per-kernel roofline analysis, memory throughput, warp efficiency |
| `nvidia-smi` | Real-time GPU utilization and memory usage |
| Triton `perf_analyzer` | Triton-specific throughput and latency measurement |

## Reports

Saved reports go in `profiling/reports/`. They are excluded from git but can be shared as build artifacts.

## Bottleneck Analysis Template

Use this structure when documenting a profiling finding:

```
**Bottleneck observed:**
  <What was slow, where it appeared in the trace>

**Evidence:**
  <Nsight metric, timeline screenshot, or perf_analyzer output>

**Fix attempted:**
  <What was changed — kernel launch config, batch size, memory layout, etc.>

**Before / after:**
  | Metric | Before | After |
  |---|---|---|
  | p50 latency | X ms | Y ms |
  | GPU utilization | X% | Y% |
  | Kernel time | X μs | Y μs |
```

See `nsight_systems_commands.md` and `nsight_compute_commands.md` for exact profiling commands.
