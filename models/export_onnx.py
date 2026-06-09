"""Export pretrained ResNet-50 to ONNX."""
import argparse
import pathlib

import torch
import torchvision.models as models


def export(
    output: pathlib.Path,
    batch_size: int = 1,
    opset: int = 17,
    device: str = "cpu",
    dynamic_batch: bool = True,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained ResNet-50 on {device}...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval().to(device)

    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    print(f"Exporting to {output} (opset {opset}, dynamic_batch={dynamic_batch})...")
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"Exported: {output} ({output.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ResNet-50 to ONNX")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("model_repository/resnet50_onnx/1/model.onnx"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--no-dynamic-batch", action="store_true")
    args = parser.parse_args()

    export(
        output=args.output,
        batch_size=args.batch_size,
        opset=args.opset,
        device=args.device,
        dynamic_batch=not args.no_dynamic_batch,
    )


if __name__ == "__main__":
    main()
