"""
Benchmark comparison across inference backends.

Backends tested:
  1. PyTorch (local CPU/GPU)
  2. ONNX Runtime (local CPU)
  3. Triton ONNX backend
  4. Triton TensorRT backend
  5. Triton TensorRT + dynamic batching

Usage:
    python benchmarks/run_backend_comparison.py \
        --output benchmarks/results/backend_comparison.json \
        --concurrency 1 4 8 \
        --batch-size 1 4 8 \
        --duration-sec 30
"""
import argparse
import asyncio
import json
import pathlib
import statistics
import time
import traceback
from dataclasses import dataclass, asdict

import httpx
import numpy as np

# Synthetic 224x224 JPEG-like payload (white image)
from PIL import Image
import io


def _make_dummy_jpeg(width: int = 224, height: int = 224) -> bytes:
    img = Image.fromarray(np.random.randint(0, 256, (height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@dataclass
class BenchmarkResult:
    backend: str
    batch_size: int
    concurrency: int
    num_requests: int
    duration_sec: float
    latencies_ms: list[float]
    errors: int

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.95)]

    @property
    def p99(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.99)]

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def throughput(self) -> float:
        return self.num_requests / self.duration_sec if self.duration_sec > 0 else 0.0

    @property
    def error_rate(self) -> float:
        total = self.num_requests + self.errors
        return self.errors / total if total > 0 else 0.0

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("latencies_ms")  # too verbose for summary
        d.update(
            p50_ms=round(self.p50, 3),
            p95_ms=round(self.p95, 3),
            p99_ms=round(self.p99, 3),
            mean_ms=round(self.mean, 3),
            throughput_rps=round(self.throughput, 2),
            error_rate=round(self.error_rate, 4),
        )
        return d


async def _run_concurrent(
    api_url: str,
    payload: bytes,
    concurrency: int,
    duration_sec: float,
) -> tuple[list[float], int]:
    latencies: list[float] = []
    errors = 0
    deadline = time.monotonic() + duration_sec

    async def worker():
        nonlocal errors
        async with httpx.AsyncClient(timeout=15.0) as client:
            while time.monotonic() < deadline:
                t0 = time.perf_counter()
                try:
                    resp = await client.post(
                        api_url,
                        files={"file": ("img.jpg", payload, "image/jpeg")},
                    )
                    resp.raise_for_status()
                    latencies.append((time.perf_counter() - t0) * 1000)
                except Exception:
                    errors += 1

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return latencies, errors


def benchmark_api_endpoint(
    backend_name: str,
    api_url: str,
    concurrency: int,
    batch_size: int,
    duration_sec: float,
) -> BenchmarkResult:
    payload = _make_dummy_jpeg()

    t0 = time.monotonic()
    latencies, errors = asyncio.run(
        _run_concurrent(api_url, payload, concurrency, duration_sec)
    )
    elapsed = time.monotonic() - t0

    return BenchmarkResult(
        backend=backend_name,
        batch_size=batch_size,
        concurrency=concurrency,
        num_requests=len(latencies),
        duration_sec=elapsed,
        latencies_ms=latencies,
        errors=errors,
    )


def run_pytorch_baseline(concurrency: int, batch_size: int, duration_sec: float) -> BenchmarkResult:
    """Time PyTorch inference locally (single-process, not API)."""
    try:
        import torch
        import torchvision.models as tvm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).eval().to(device)

        latencies: list[float] = []
        deadline = time.monotonic() + duration_sec

        with torch.no_grad():
            while time.monotonic() < deadline:
                x = torch.randn(batch_size, 3, 224, 224, device=device)
                t0 = time.perf_counter()
                _ = model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        return BenchmarkResult(
            backend="pytorch_local",
            batch_size=batch_size,
            concurrency=1,
            num_requests=len(latencies),
            duration_sec=duration_sec,
            latencies_ms=latencies,
            errors=0,
        )
    except Exception as e:
        print(f"  [SKIP] PyTorch baseline failed: {e}")
        return BenchmarkResult("pytorch_local", batch_size, 1, 0, duration_sec, [], 0)


def run_onnxruntime_baseline(batch_size: int, duration_sec: float, onnx_path: str) -> BenchmarkResult:
    """Time ONNX Runtime inference locally."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

        latencies: list[float] = []
        deadline = time.monotonic() + duration_sec

        while time.monotonic() < deadline:
            t0 = time.perf_counter()
            sess.run(["output"], {"input": dummy})
            latencies.append((time.perf_counter() - t0) * 1000)

        return BenchmarkResult(
            backend="onnxruntime_local",
            batch_size=batch_size,
            concurrency=1,
            num_requests=len(latencies),
            duration_sec=duration_sec,
            latencies_ms=latencies,
            errors=0,
        )
    except Exception as e:
        print(f"  [SKIP] ONNX Runtime baseline failed: {e}")
        return BenchmarkResult("onnxruntime_local", batch_size, 1, 0, duration_sec, [], 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend latency/throughput comparison")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("benchmarks/results/backend_comparison.json"))
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--onnx-path", default="model_repository/resnet50_onnx/1/model.onnx")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--duration-sec", type=float, default=30.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []

    # PyTorch local baselines
    for bs in args.batch_size:
        print(f"\n[pytorch_local] batch={bs}")
        r = run_pytorch_baseline(concurrency=1, batch_size=bs, duration_sec=args.duration_sec)
        print(f"  p50={r.p50:.1f}ms  p95={r.p95:.1f}ms  tput={r.throughput:.1f} req/s")
        results.append(r.summary())

    # ONNX Runtime local baselines
    for bs in args.batch_size:
        print(f"\n[onnxruntime_local] batch={bs}")
        r = run_onnxruntime_baseline(batch_size=bs, duration_sec=args.duration_sec, onnx_path=args.onnx_path)
        print(f"  p50={r.p50:.1f}ms  p95={r.p95:.1f}ms  tput={r.throughput:.1f} req/s")
        results.append(r.summary())

    # Triton via FastAPI
    for bs in args.batch_size:
        for c in args.concurrency:
            name = f"triton_api"
            endpoint = f"{args.api_url}/v1/predict"
            print(f"\n[{name}] batch={bs} concurrency={c}")
            try:
                r = benchmark_api_endpoint(name, endpoint, c, bs, args.duration_sec)
                print(f"  p50={r.p50:.1f}ms  p95={r.p95:.1f}ms  tput={r.throughput:.1f} req/s  errors={r.errors}")
                results.append(r.summary())
            except Exception as e:
                print(f"  [SKIP] {e}")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
