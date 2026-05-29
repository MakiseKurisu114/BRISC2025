import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-brisc")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BriscSegmentationDataset
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
    parser = argparse.ArgumentParser(description="Run final BRISC segmentation inference and visualizations")
    parser.add_argument("--final-model-json", type=Path, default=Path("outputs/final_model/final_model.json"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final_model"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_final_model(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    final_model = data.get("final_model")
    if not final_model:
        raise KeyError(f"Missing final_model in {path}")
    required = ["name", "method", "checkpoint", "threshold"]
    missing = [key for key in required if key not in final_model]
    if missing:
        raise KeyError(f"Missing final_model fields in {path}: {missing}")
    return final_model


def build_model(method: str, base_channels: int) -> torch.nn.Module:
    method_lower = method.lower()
    if "attention" in method_lower:
        return AttentionUNet(in_channels=1, out_channels=1, base_channels=base_channels)
    return UNet(in_channels=1, out_channels=1, base_channels=base_channels)


def parse_name(name: str) -> tuple[str, str]:
    parts = name.split("_")
    tumor = TUMOR_NAMES.get(parts[3], parts[3]) if len(parts) > 3 else "unknown"
    view = VIEW_NAMES.get(parts[4], parts[4]) if len(parts) > 4 else "unknown"
    return tumor, view


def size_group(mask: torch.Tensor) -> tuple[float, str, str]:
    area_ratio = mask.float().mean().item()
    if area_ratio < 0.01:
        return area_ratio, "small", "small < 1%"
    if area_ratio <= 0.05:
        return area_ratio, "medium", "medium 1%-5%"
    return area_ratio, "large", "large > 5%"


def binary_metrics(prob: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[float, float, float, float]:
    pred = (prob > threshold).float().flatten()
    target = (target > 0.5).float().flatten()
    tp = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = pred_sum + target_sum - tp
    eps = 1e-7
    dice = (2.0 * tp + eps) / (pred_sum + target_sum + eps)
    iou = (tp + eps) / (union + eps)
    precision = (tp + eps) / (pred_sum + eps)
    recall = (tp + eps) / (target_sum + eps)
    return dice.item(), iou.item(), precision.item(), recall.item()


@torch.no_grad()
def collect_rows(model, loader, device, threshold: float) -> list[dict]:
    rows = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        for i, name in enumerate(batch["name"]):
            dice, iou, precision, recall = binary_metrics(probs[i], masks[i], threshold)
            tumor, view = parse_name(name)
            area_ratio, size_key, size_label = size_group(masks[i])
            rows.append(
                {
                    "name": name,
                    "tumor": tumor,
                    "view": view,
                    "area_ratio": area_ratio,
                    "size_group": size_key,
                    "size_label": size_label,
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "image": images[i].detach().cpu(),
                    "mask": masks[i].detach().cpu(),
                    "pred": preds[i].detach().cpu(),
                }
            )
    return rows


def mean_value(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def grouped_metrics(rows: list[dict]) -> list[dict]:
    group_specs = [
        ("tumor", "tumor", ["glioma", "meningioma", "pituitary"]),
        ("view", "view", ["axial", "coronal", "sagittal"]),
        ("size", "size_group", ["small", "medium", "large"]),
    ]
    output = []
    for group_type, key, group_names in group_specs:
        for group_name in group_names:
            group_rows = [row for row in rows if row[key] == group_name]
            if not group_rows:
                continue
            label = group_rows[0]["size_label"] if group_type == "size" else group_name
            output.append(
                {
                    "group_type": group_type,
                    "group": group_name,
                    "group_label": label,
                    "n": len(group_rows),
                    "mean_dice": mean_value(group_rows, "dice"),
                    "mean_iou": mean_value(group_rows, "iou"),
                    "mean_precision": mean_value(group_rows, "precision"),
                    "mean_recall": mean_value(group_rows, "recall"),
                    "mean_area_ratio": mean_value(group_rows, "area_ratio"),
                }
            )
    return output


def select_representatives(rows: list[dict], prefix: str, group_type: str, group_label: str) -> list[dict]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: row["dice"])
    median_dice = float(np.median([row["dice"] for row in sorted_rows]))
    selected = [
        ("worst", sorted_rows[0]),
        ("typical", min(sorted_rows, key=lambda row: abs(row["dice"] - median_dice))),
        ("best", sorted_rows[-1]),
    ]
    seen = set()
    output = []
    for kind, row in selected:
        key = row["name"]
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["selection"] = kind
        item["figure"] = prefix
        item["group_type"] = group_type
        item["group_label"] = group_label
        output.append(item)
    return output


def select_overall(rows: list[dict]) -> list[dict]:
    sorted_rows = sorted(rows, key=lambda row: row["dice"])
    median_dice = float(np.median([row["dice"] for row in sorted_rows]))
    worst = sorted_rows[:2]
    best = sorted_rows[-2:]
    typical = sorted(sorted_rows, key=lambda row: abs(row["dice"] - median_dice))[:2]
    selected = [("worst", row) for row in worst] + [("typical", row) for row in typical] + [("best", row) for row in best]
    seen = set()
    output = []
    for kind, row in selected:
        if row["name"] in seen:
            continue
        seen.add(row["name"])
        item = dict(row)
        item["selection"] = kind
        item["figure"] = "overall_examples.png"
        item["group_type"] = "overall"
        item["group_label"] = "overall"
        output.append(item)
    return output


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().numpy()


def save_grid(rows: list[dict], out_path: Path, title: str) -> None:
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3.2 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for ax_row, row in zip(axes, rows):
        image = to_numpy(row["image"])
        mask = to_numpy(row["mask"])
        pred = to_numpy(row["pred"])

        ax_row[0].imshow(image, cmap="gray")
        ax_row[0].set_title(f"{row['selection']}: {row['name']}")

        ax_row[1].imshow(mask, cmap="gray")
        ax_row[1].set_title("Ground truth")

        ax_row[2].imshow(pred, cmap="gray")
        ax_row[2].set_title(f"Prediction\nDice={row['dice']:.3f} IoU={row['iou']:.3f}")

        ax_row[3].imshow(image, cmap="gray")
        if mask.max() > 0:
            ax_row[3].contour(mask, levels=[0.5], colors=["lime"], linewidths=1.0)
        if pred.max() > 0:
            ax_row[3].contour(pred, levels=[0.5], colors=["red"], linewidths=1.0)
        ax_row[3].set_title("Overlay\nGT=green Pred=red")

        for ax in ax_row:
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_per_sample(rows: list[dict], out_path: Path) -> None:
    fields = [
        "name",
        "tumor",
        "view",
        "area_ratio",
        "size_group",
        "size_label",
        "dice",
        "iou",
        "precision",
        "recall",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["name"]):
            writer.writerow({field: row[field] for field in fields})


def write_group_metrics(group_rows: list[dict], out_path: Path) -> None:
    fields = [
        "group_type",
        "group",
        "group_label",
        "n",
        "mean_dice",
        "mean_iou",
        "mean_precision",
        "mean_recall",
        "mean_area_ratio",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(group_rows)


def write_summary(final_model: dict, rows: list[dict], out_path: Path) -> None:
    fields = ["split", "final_model", "method", "checkpoint", "threshold", "n_samples", "test_dice", "test_iou", "test_precision", "test_recall"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "split": "segmentation_task/test",
                "final_model": final_model["name"],
                "method": final_model["method"],
                "checkpoint": final_model["checkpoint"],
                "threshold": final_model["threshold"],
                "n_samples": len(rows),
                "test_dice": mean_value(rows, "dice"),
                "test_iou": mean_value(rows, "iou"),
                "test_precision": mean_value(rows, "precision"),
                "test_recall": mean_value(rows, "recall"),
            }
        )


def write_selected_examples(rows: list[dict], out_path: Path) -> None:
    fields = ["figure", "group_type", "group_label", "selection", "name", "tumor", "view", "size_label", "dice", "iou", "precision", "recall"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    final_model = load_final_model(args.final_model_json)
    checkpoint_path = Path(final_model["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Final model checkpoint not found: {checkpoint_path}")

    threshold = float(final_model["threshold"])
    dataset = BriscSegmentationDataset(args.data_root, split="test", image_size=args.image_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(final_model["method"], args.base_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    rows = collect_rows(model, loader, device, threshold=threshold)
    group_rows = grouped_metrics(rows)

    write_per_sample(rows, args.out_dir / "per_sample_metrics.csv")
    write_group_metrics(group_rows, args.out_dir / "group_metrics.csv")
    write_summary(final_model, rows, args.out_dir / "final_test_summary.csv")

    selected_rows = []
    overall_rows = select_overall(rows)
    save_grid(overall_rows, figures_dir / "overall_examples.png", "Final model overall examples")
    selected_rows.extend(overall_rows)

    figure_specs = [
        ("tumor", "tumor", "glioma", "tumor_glioma_examples.png"),
        ("tumor", "tumor", "meningioma", "tumor_meningioma_examples.png"),
        ("tumor", "tumor", "pituitary", "tumor_pituitary_examples.png"),
        ("view", "view", "axial", "view_axial_examples.png"),
        ("view", "view", "coronal", "view_coronal_examples.png"),
        ("view", "view", "sagittal", "view_sagittal_examples.png"),
        ("size", "size_group", "small", "size_small_examples.png"),
        ("size", "size_group", "medium", "size_medium_examples.png"),
        ("size", "size_group", "large", "size_large_examples.png"),
    ]
    for group_type, key, value, filename in figure_specs:
        group = [row for row in rows if row[key] == value]
        label = group[0]["size_label"] if group_type == "size" and group else value
        examples = select_representatives(group, filename, group_type, label)
        save_grid(examples, figures_dir / filename, f"Final model {group_type}: {label}")
        selected_rows.extend(examples)

    write_selected_examples(selected_rows, args.out_dir / "selected_examples.csv")

    print(f"device={device}")
    print(f"split=segmentation_task/test samples={len(rows)}")
    print(f"checkpoint={checkpoint_path}")
    print(f"threshold={threshold}")
    print(f"test_dice={mean_value(rows, 'dice'):.4f} test_iou={mean_value(rows, 'iou'):.4f}")
    print(f"saved={args.out_dir / 'per_sample_metrics.csv'}")
    print(f"saved={args.out_dir / 'group_metrics.csv'}")
    print(f"saved={args.out_dir / 'final_test_summary.csv'}")
    print(f"saved={args.out_dir / 'selected_examples.csv'}")
    print(f"saved_figures={figures_dir}")


if __name__ == "__main__":
    main()
