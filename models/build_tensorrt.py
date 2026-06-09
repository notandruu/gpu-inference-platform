"""Build a TensorRT engine from the ONNX model via Python API or trtexec fallback."""
import argparse
import pathlib
import subprocess
import sys
import time


def build_with_trtexec(
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    fp16: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    log_path: pathlib.Path,
) -> None:
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=input:{min_batch}x3x224x224",
        f"--optShapes=input:{opt_batch}x3x224x224",
        f"--maxShapes=input:{max_batch}x3x224x224",
    ]
    if fp16:
        cmd.append("--fp16")

    print("Running trtexec:", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        print(f"trtexec failed. See log: {log_path}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"Engine saved: {engine_path}")
    print(f"Build log: {log_path}")


def build_with_python_api(
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    fp16: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    log_path: pathlib.Path,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise RuntimeError("Failed to parse ONNX model")

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (min_batch, 3, 224, 224), (opt_batch, 3, 224, 224), (max_batch, 3, 224, 224))
    config.add_optimization_profile(profile)

    print("Building TensorRT engine (this may take several minutes)...")
    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    elapsed = time.time() - t0

    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_bytes)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"Build time: {elapsed:.1f}s\nFP16: {fp16}\n")

    print(f"Engine saved: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Build time: {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument(
        "--onnx",
        type=pathlib.Path,
        default=pathlib.Path("model_repository/resnet50_onnx/1/model.onnx"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("model_repository/resnet50_trt/1/model.plan"),
    )
    parser.add_argument(
        "--log",
        type=pathlib.Path,
        default=pathlib.Path("benchmarks/results/tensorrt_build.log"),
    )
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--use-trtexec", action="store_true")
    args = parser.parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}\nRun: make export-onnx")

    if args.use_trtexec:
        build_with_trtexec(
            args.onnx, args.output, args.fp16,
            args.min_batch, args.opt_batch, args.max_batch, args.log,
        )
    else:
        try:
            build_with_python_api(
                args.onnx, args.output, args.fp16,
                args.min_batch, args.opt_batch, args.max_batch, args.log,
            )
        except ImportError:
            print("TensorRT Python package not found; falling back to trtexec")
            build_with_trtexec(
                args.onnx, args.output, args.fp16,
                args.min_batch, args.opt_batch, args.max_batch, args.log,
            )


if __name__ == "__main__":
    main()
