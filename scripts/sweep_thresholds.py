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
    parser = argparse.ArgumentParser(
        description=(
            "Sweep probability thresholds for BRISC segmentation checkpoints. "
            "Default split is validation; test should be used only after the final "
            "configuration is fixed, not for threshold selection."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["unet", "attention_unet"], required=True)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--out-dir", type=Path, required=True)
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


def dice_iou_from_prob(prob: torch.Tensor, target: torch.Tensor, threshold: float, eps: float = 1e-7) -> tuple[float, float]:
    pred = (prob > threshold).float()
    pred = pred.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


@torch.no_grad()
def collect_rows(model, loader, device, thresholds: list[float]) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        probs = torch.sigmoid(model(images))

        for i, name in enumerate(batch["name"]):
            tumor, view = parse_name(name)
            area_ratio, area_group = size_group(masks[i])
            for threshold in thresholds:
                dice, iou = dice_iou_from_prob(probs[i : i + 1], masks[i : i + 1], threshold)
                rows.append(
                    {
                        "threshold": threshold,
                        "name": name,
                        "tumor": tumor,
                        "view": view,
                        "area_ratio": area_ratio,
                        "size_group": area_group,
                        "dice": dice,
                        "iou": iou,
                    }
                )
    return rows


def grouped_metrics(rows: list[dict], key: str) -> list[dict]:
    stats = []
    thresholds = sorted({row["threshold"] for row in rows})
    for threshold in thresholds:
        threshold_rows = [row for row in rows if row["threshold"] == threshold]
        groups = {}
        for row in threshold_rows:
            groups.setdefault(row[key], []).append(row)
        for group_name, group_rows in sorted(groups.items()):
            n = len(group_rows)
            stats.append(
                {
                    "threshold": threshold,
                    "group_type": key,
                    "group": group_name,
                    "n": n,
                    "mean_dice": sum(row["dice"] for row in group_rows) / n,
                    "mean_iou": sum(row["iou"] for row in group_rows) / n,
                }
            )
    return stats


def save_overall(rows: list[dict], out_path: Path) -> None:
    thresholds = sorted({row["threshold"] for row in rows})
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "mean_dice", "mean_iou", "n_samples"])
        for threshold in thresholds:
            threshold_rows = [row for row in rows if row["threshold"] == threshold]
            writer.writerow(
                [
                    threshold,
                    sum(row["dice"] for row in threshold_rows) / len(threshold_rows),
                    sum(row["iou"] for row in threshold_rows) / len(threshold_rows),
                    len(threshold_rows),
                ]
            )


def save_group_metrics(rows: list[dict], out_path: Path) -> None:
    group_rows = []
    for key in ["tumor", "view", "size_group"]:
        group_rows.extend(grouped_metrics(rows, key))
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "group_type", "group", "n", "mean_dice", "mean_iou"])
        for row in group_rows:
            writer.writerow([row["threshold"], row["group_type"], row["group"], row["n"], row["mean_dice"], row["mean_iou"]])


def save_small_plot(group_metrics_path: Path, out_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(group_metrics_path)
    small = df[(df["group_type"] == "size_group") & (df["group"] == "small_<1%")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(small["threshold"], small["mean_dice"], marker="o", label="small Dice")
    ax.plot(small["threshold"], small["mean_iou"], marker="o", label="small IoU")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.55, 0.85)
    ax.set_title("Small Tumor Metrics by Threshold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval_split == "val":
        _, dataset = make_train_val_datasets(
            args.data_root,
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        dataset = BriscSegmentationDataset(args.data_root, split="test", image_size=args.image_size, augment=False)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.model, args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    rows = collect_rows(model, loader, device, args.thresholds)

    overall_path = args.out_dir / "threshold_overall_metrics.csv"
    group_path = args.out_dir / "threshold_group_metrics.csv"
    save_overall(rows, overall_path)
    save_group_metrics(rows, group_path)
    save_small_plot(group_path, args.out_dir / "small_tumor_threshold_curve.png")

    print(f"device={device}")
    print(f"eval_split={args.eval_split} samples={len(dataset)} thresholds={args.thresholds}")
    print(f"saved: {overall_path}")
    print(f"saved: {group_path}")
    print(f"saved: {args.out_dir / 'small_tumor_threshold_curve.png'}")


if __name__ == "__main__":
    main()
