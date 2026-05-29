import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-brisc")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.dataset import BriscSegmentationDataset, make_train_val_datasets
from src.metrics import dice_iou_from_logits
from src.model_attention_unet import AttentionUNet
from src.model_unet import UNet


TUMOR_NAMES = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "no": "no_tumor",
    "nt": "no_tumor",
}

VIEW_NAMES = {
    "ax": "axial",
    "co": "coronal",
    "sa": "sagittal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved BRISC segmentation checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["unet", "attention_unet"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def build_model(name: str, base_channels: int) -> torch.nn.Module:
    if name == "unet":
        return UNet(in_channels=1, out_channels=1, base_channels=base_channels)
    return AttentionUNet(in_channels=1, out_channels=1, base_channels=base_channels)


def parse_name(name: str) -> tuple[str, str]:
    parts = name.split("_")
    tumor = TUMOR_NAMES.get(parts[3], parts[3]) if len(parts) > 4 else "unknown"
    view = VIEW_NAMES.get(parts[4], parts[4]) if len(parts) > 5 else "unknown"
    return tumor, view


def size_group(mask: torch.Tensor) -> tuple[float, str]:
    area_ratio = mask.float().mean().item()
    if area_ratio < 0.01:
        return area_ratio, "small_<1%"
    if area_ratio <= 0.05:
        return area_ratio, "medium_1-5%"
    return area_ratio, "large_>5%"


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        for i, name in enumerate(batch["name"]):
            sample_dice, sample_iou = dice_iou_from_logits(logits[i : i + 1], masks[i : i + 1], threshold=threshold)
            tumor, view = parse_name(name)
            area_ratio, area_group = size_group(masks[i])
            rows.append(
                {
                    "name": name,
                    "tumor": tumor,
                    "view": view,
                    "area_ratio": area_ratio,
                    "size_group": area_group,
                    "dice": sample_dice,
                    "iou": sample_iou,
                    "image": images[i].detach().cpu(),
                    "mask": masks[i].detach().cpu(),
                    "prob": probs[i].detach().cpu(),
                    "pred": preds[i].detach().cpu(),
                }
            )
    mean_dice = sum(row["dice"] for row in rows) / len(rows)
    mean_iou = sum(row["iou"] for row in rows) / len(rows)
    return mean_dice, mean_iou, rows


def grouped_metrics(rows, key: str) -> list[dict[str, float | int | str]]:
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)

    stats = []
    for group_name, group_rows in sorted(groups.items()):
        n = len(group_rows)
        stats.append(
            {
                "group_type": key,
                "group": group_name,
                "n": n,
                "mean_dice": sum(row["dice"] for row in group_rows) / n,
                "mean_iou": sum(row["iou"] for row in group_rows) / n,
            }
        )
    return stats


def save_visual_grid(rows, out_path: Path, max_visuals: int) -> None:
    rows = sorted(rows, key=lambda x: x["dice"])
    if len(rows) <= max_visuals:
        selected = rows
    else:
        selected = rows[: max_visuals // 2] + rows[-(max_visuals - max_visuals // 2) :]

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax_row, row in zip(axes, selected):
        ax_row[0].imshow(row["image"].squeeze(), cmap="gray")
        ax_row[0].set_title(f"MRI\n{row['name']}")
        ax_row[1].imshow(row["mask"].squeeze(), cmap="gray")
        ax_row[1].set_title("GT mask")
        ax_row[2].imshow(row["prob"].squeeze(), cmap="gray")
        ax_row[2].set_title("Pred prob")
        ax_row[3].imshow(row["pred"].squeeze(), cmap="gray")
        ax_row[3].set_title(f"Pred mask\nDice={row['dice']:.3f}")
        for ax in ax_row:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval_split == "val":
        _, eval_dataset = make_train_val_datasets(
            args.data_root,
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        eval_dataset = BriscSegmentationDataset(
            args.data_root,
            split="test",
            image_size=args.image_size,
            augment=False,
        )
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    mean_dice, mean_iou, rows = evaluate(model, loader, device, threshold=args.threshold)
    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_split", "threshold", "mean_dice", "mean_iou", "n_samples"])
        writer.writerow([args.eval_split, args.threshold, mean_dice, mean_iou, len(eval_dataset)])

    per_sample_path = args.out_dir / "per_sample_metrics.csv"
    with per_sample_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "tumor", "view", "area_ratio", "size_group", "dice", "iou"])
        for row in sorted(rows, key=lambda x: x["dice"]):
            writer.writerow(
                [
                    row["name"],
                    row["tumor"],
                    row["view"],
                    row["area_ratio"],
                    row["size_group"],
                    row["dice"],
                    row["iou"],
                ]
            )

    group_metrics_path = args.out_dir / "group_metrics.csv"
    group_rows = []
    for key in ["tumor", "view", "size_group"]:
        group_rows.extend(grouped_metrics(rows, key))
    with group_metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_type", "group", "n", "mean_dice", "mean_iou"])
        for row in group_rows:
            writer.writerow([row["group_type"], row["group"], row["n"], row["mean_dice"], row["mean_iou"]])

    visual_path = args.out_dir / f"{args.eval_split}_examples_worst_best.png"
    save_visual_grid(rows, visual_path, args.max_visuals)

    print(f"device={device}")
    print(f"eval_split={args.eval_split} threshold={args.threshold} samples={len(eval_dataset)}")
    print(f"mean_dice={mean_dice:.4f} mean_iou={mean_iou:.4f}")
    print(f"saved: {metrics_path}")
    print(f"saved: {per_sample_path}")
    print(f"saved: {group_metrics_path}")
    print(f"saved: {visual_path}")


if __name__ == "__main__":
    main()
