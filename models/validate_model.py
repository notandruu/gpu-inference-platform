"""Compare PyTorch ResNet-50 output against ONNX Runtime output."""
import argparse
import pathlib

import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort


def validate(onnx_path: pathlib.Path, device: str = "cpu", num_runs: int = 5) -> None:
    print(f"Loading ONNX model from {onnx_path}...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    print("Loading PyTorch ResNet-50...")
    pt_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    pt_model.eval().to(device)

    max_diffs = []
    for i in range(num_runs):
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            pt_out = pt_model(dummy.to(device)).cpu().numpy()

        onnx_out = sess.run(["output"], {"input": dummy.numpy()})[0]
        diff = np.abs(pt_out - onnx_out).max()
        max_diffs.append(diff)
        print(f"  run {i+1}: max |PT - ONNX| = {diff:.6f}")

    overall = max(max_diffs)
    status = "PASS" if overall < 1e-3 else "WARN"
    print(f"\n[{status}] max difference across {num_runs} runs: {overall:.6f}")
    if overall >= 1e-3:
        print("  WARNING: difference exceeds 1e-3 — check opset or precision settings")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ONNX model against PyTorch")
    parser.add_argument(
        "--onnx",
        type=pathlib.Path,
        default=pathlib.Path("model_repository/resnet50_onnx/1/model.onnx"),
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-runs", type=int, default=5)
    args = parser.parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}\nRun: make export-onnx")

    validate(args.onnx, device=args.device, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
