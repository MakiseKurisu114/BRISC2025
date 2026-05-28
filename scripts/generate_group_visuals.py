import argparse
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
    parser = argparse.ArgumentParser(description="Generate grouped visual examples for BRISC segmentation results")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["unet", "attention_unet"], required=True)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples-per-group", type=int, default=4)
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


def size_group(mask: torch.Tensor) -> str:
    area_ratio = mask.float().mean().item()
    if area_ratio < 0.01:
        return "small_lt1pct"
    if area_ratio <= 0.05:
        return "medium_1to5pct"
    return "large_gt5pct"


@torch.no_grad()
def collect_rows(model, loader, device) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        for i, name in enumerate(batch["name"]):
            dice, iou = dice_iou_from_logits(logits[i : i + 1], masks[i : i + 1])
            tumor, view = parse_name(name)
            rows.append(
                {
                    "name": name,
                    "tumor": tumor,
                    "view": view,
                    "size": size_group(masks[i]),
                    "dice": dice,
                    "iou": iou,
                    "image": images[i].detach().cpu(),
                    "mask": masks[i].detach().cpu(),
                    "prob": probs[i].detach().cpu(),
                    "pred": preds[i].detach().cpu(),
                }
            )
    return rows


def select_examples(rows: list[dict], n: int) -> list[dict]:
    rows = sorted(rows, key=lambda row: row["dice"])
    if len(rows) <= n:
        return rows
    n_worst = n // 2
    n_best = n - n_worst
    return rows[:n_worst] + rows[-n_best:]


def save_grid(rows: list[dict], out_path: Path, title: str) -> None:
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for ax_row, row in zip(axes, rows):
        ax_row[0].imshow(row["image"].squeeze(), cmap="gray")
        ax_row[0].set_title(f"MRI\n{row['name']}")
        ax_row[1].imshow(row["mask"].squeeze(), cmap="gray")
        ax_row[1].set_title("GT mask")
        ax_row[2].imshow(row["prob"].squeeze(), cmap="gray")
        ax_row[2].set_title("Pred prob")
        ax_row[3].imshow(row["pred"].squeeze(), cmap="gray")
        ax_row[3].set_title(f"Pred mask\nDice={row['dice']:.3f} IoU={row['iou']:.3f}")
        for ax in ax_row:
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_split == "val":
        _, dataset = make_train_val_datasets(
            args.data_root,
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        dataset = BriscSegmentationDataset(
            args.data_root,
            split="test",
            image_size=args.image_size,
            augment=False,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.base_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = collect_rows(model, loader, device)

    group_specs = [
        ("tumor", ["glioma", "meningioma", "pituitary"]),
        ("view", ["axial", "coronal", "sagittal"]),
        ("size", ["small_lt1pct", "medium_1to5pct", "large_gt5pct"]),
    ]

    saved = []
    for group_type, group_names in group_specs:
        for group_name in group_names:
            group_rows = [row for row in rows if row[group_type] == group_name]
            examples = select_examples(group_rows, args.examples_per_group)
            out_path = args.out_dir / f"{args.eval_split}_{group_type}_{group_name}_worst_best.png"
            title = f"{args.eval_split.upper()} {group_type}: {group_name} worst/best examples"
            save_grid(examples, out_path, title)
            saved.append(out_path)

    print(f"device={device}")
    print(f"eval_split={args.eval_split} samples={len(dataset)}")
    for path in saved:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
