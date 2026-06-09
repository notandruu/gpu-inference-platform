"""
Benchmark CUDA preprocessing extension vs NumPy/PyTorch baselines.

Run:
    python benchmarks/run_preprocess_benchmark.py

GPU is required for the CUDA path; NumPy/PyTorch paths run on CPU.
"""
import json
import pathlib
import statistics
import time

import numpy as np


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def numpy_preprocess(images: np.ndarray) -> np.ndarray:
    arr = images.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(0, 3, 1, 2)


def torch_preprocess(images: np.ndarray) -> "np.ndarray":
    import torch
    t = torch.from_numpy(images).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
    t = (t - mean) / std
    return t.permute(0, 3, 1, 2).numpy()


def cuda_preprocess(images: np.ndarray) -> np.ndarray:
    import cuda_preprocess as cp
    return cp.preprocess(images)


def bench(fn, images, warmup=5, runs=50) -> list[float]:
    for _ in range(warmup):
        fn(images)
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(images)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def main():
    output_path = pathlib.Path("benchmarks/results/preprocess_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    batch_sizes = [1, 4, 8, 16, 32]

    for bs in batch_sizes:
        images = np.random.randint(0, 256, (bs, 224, 224, 3), dtype=np.uint8)
        print(f"\nbatch_size={bs}")

        for name, fn in [("numpy", numpy_preprocess), ("torch", torch_preprocess)]:
            try:
                lats = bench(fn, images)
                p50 = statistics.median(lats)
                print(f"  [{name}] p50={p50:.3f}ms  mean={statistics.mean(lats):.3f}ms")
                results.append({"backend": name, "batch_size": bs, "p50_ms": round(p50, 4), "mean_ms": round(statistics.mean(lats), 4)})
            except Exception as e:
                print(f"  [{name}] SKIP: {e}")

        try:
            lats = bench(cuda_preprocess, images)
            p50 = statistics.median(lats)
            print(f"  [cuda]  p50={p50:.3f}ms  mean={statistics.mean(lats):.3f}ms")
            results.append({"backend": "cuda", "batch_size": bs, "p50_ms": round(p50, 4), "mean_ms": round(statistics.mean(lats), 4)})
        except ImportError:
            print("  [cuda]  SKIP: extension not built (make build-cuda)")

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
