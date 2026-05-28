import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.model_attention_unet import AttentionUNet
from src.model_unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export readable information from a PyTorch checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["unet", "attention_unet"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=16)
    return parser.parse_args()


def build_model(name: str, base_channels: int) -> torch.nn.Module:
    if name == "unet":
        return UNet(in_channels=1, out_channels=1, base_channels=base_channels)
    return AttentionUNet(in_channels=1, out_channels=1, base_channels=base_channels)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    model = build_model(args.model, args.base_channels)
    model.load_state_dict(state_dict)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    summary = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "best_epoch": checkpoint.get("epoch"),
        "saved_val_dice": checkpoint.get("val_dice"),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "args": json_safe(checkpoint.get("args", {})),
    }

    with (args.out_dir / "checkpoint_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (args.out_dir / "model_architecture.txt").open("w", encoding="utf-8") as f:
        f.write(str(model))
        f.write("\n")

    with (args.out_dir / "parameter_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "shape", "num_params", "mean", "std", "min", "max"])
        for name, tensor in state_dict.items():
            tensor = tensor.detach().float()
            writer.writerow(
                [
                    name,
                    list(tensor.shape),
                    tensor.numel(),
                    tensor.mean().item() if tensor.numel() else 0.0,
                    tensor.std(unbiased=False).item() if tensor.numel() else 0.0,
                    tensor.min().item() if tensor.numel() else 0.0,
                    tensor.max().item() if tensor.numel() else 0.0,
                ]
            )

    print(f"saved: {args.out_dir / 'checkpoint_summary.json'}")
    print(f"saved: {args.out_dir / 'model_architecture.txt'}")
    print(f"saved: {args.out_dir / 'parameter_summary.csv'}")


if __name__ == "__main__":
    main()
