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
import torch
from torch.utils.data import DataLoader

from src.dataset import BriscSegmentationDataset, make_train_val_datasets
from src.model_unet import UNet


TUMOR_NAMES = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "no_tumor", "nt": "no_tumor"}
VIEW_NAMES = {"ax": "axial", "co": "coronal", "sa": "sagittal"}


CANDIDATES = [
    ("A3_original_boundary_w02", "outputs/a3/full/checkpoints/best_unet_boundary.pt", 128, 0.2, "", "bce_dice_boundary"),
    ("A3_image_size_192", "outputs/a3_tuning/image_size_192/full/checkpoints/best_unet_boundary.pt", 192, 0.2, "", "bce_dice_boundary"),
    ("A3_image_size_256_bs8", "outputs/a3_tuning/image_size_256_bs8/full/checkpoints/best_unet_boundary.pt", 256, 0.2, "", "bce_dice_boundary"),
    ("A3_small_oversampling_w15", "outputs/a3_tuning/small_oversampling_w15/full/checkpoints/best_unet_boundary.pt", 128, 0.2, "1.5", "bce_dice_boundary"),
    ("A3_small_oversampling_w2", "outputs/a3_tuning/small_oversampling_w2/full/checkpoints/best_unet_boundary.pt", 128, 0.2, "2.0", "bce_dice_boundary"),
    ("A3_small_oversampling_w3", "outputs/a3_tuning/small_oversampling_w3/full/checkpoints/best_unet_boundary.pt", 128, 0.2, "3.0", "bce_dice_boundary"),
    ("A3_focal_tversky_w02", "outputs/a3_tuning/focal_tversky_w02/full/checkpoints/best_unet_boundary_focal_tversky.pt", 128, 0.2, "", "bce_dice_boundary_focal_tversky_w02"),
    ("A3_boundary_w005", "outputs/a3_tuning/boundary_w005/full/checkpoints/best_unet_boundary.pt", 128, 0.05, "", "bce_dice_boundary"),
    ("A3_boundary_w01", "outputs/a3_tuning/boundary_w01/full/checkpoints/best_unet_boundary.pt", 128, 0.1, "", "bce_dice_boundary"),
    ("A3_boundary_w03", "outputs/a3_tuning/boundary_w03/full/checkpoints/best_unet_boundary.pt", 128, 0.3, "", "bce_dice_boundary"),
    ("A3_boundary_w05", "outputs/a3_tuning/boundary_w05/full/checkpoints/best_unet_boundary.pt", 128, 0.5, "", "bce_dice_boundary"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3 tuning validation-based selection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    val = subparsers.add_parser("val-eval")
    val.add_argument("--data-root", type=Path, default=Path("."))
    val.add_argument("--out-dir", type=Path, default=Path("outputs/a3_tuning"))
    val.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    val.add_argument("--batch-size", type=int, default=8)
    val.add_argument("--base-channels", type=int, default=16)
    val.add_argument("--val-ratio", type=float, default=0.2)
    val.add_argument("--seed", type=int, default=42)

    test = subparsers.add_parser("final-test")
    test.add_argument("--data-root", type=Path, default=Path("."))
    test.add_argument("--out-dir", type=Path, default=Path("outputs/a3_tuning"))
    test.add_argument("--selection-json", type=Path, default=Path("outputs/a3_tuning/final_selection.json"))
    test.add_argument("--batch-size", type=int, default=8)
    test.add_argument("--base-channels", type=int, default=16)
    test.add_argument("--max-visuals", type=int, default=8)
    return parser.parse_args()


def parse_name(name: str) -> tuple[str, str]:
    parts = name.split("_")
    tumor = TUMOR_NAMES.get(parts[3], parts[3]) if len(parts) > 3 else "unknown"
    view = VIEW_NAMES.get(parts[4], parts[4]) if len(parts) > 4 else "unknown"
    return tumor, view


def size_group(mask: torch.Tensor) -> tuple[float, str]:
    area_ratio = mask.float().mean().item()
    if area_ratio < 0.01:
        return area_ratio, "small_<1%"
    if area_ratio <= 0.05:
        return area_ratio, "medium_1-5%"
    return area_ratio, "large_>5%"


def sample_metrics(prob: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[float, float, float, float]:
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


def mean_rows(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n_samples": n,
        "val_dice": sum(row["dice"] for row in rows) / n,
        "val_iou": sum(row["iou"] for row in rows) / n,
        "val_precision": sum(row["precision"] for row in rows) / n,
        "val_recall": sum(row["recall"] for row in rows) / n,
    }


def mean_test_rows(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n_samples": n,
        "test_dice": sum(row["dice"] for row in rows) / n,
        "test_iou": sum(row["iou"] for row in rows) / n,
        "test_precision": sum(row["precision"] for row in rows) / n,
        "test_recall": sum(row["recall"] for row in rows) / n,
    }


def build_model(base_channels: int) -> torch.nn.Module:
    return UNet(in_channels=1, out_channels=1, base_channels=base_channels)


@torch.no_grad()
def collect_rows(candidate: dict, dataset, thresholds: list[float], device, batch_size: int, base_channels: int) -> list[dict]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(base_channels).to(device)
    checkpoint = torch.load(candidate["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        probs = torch.sigmoid(model(images))
        for i, name in enumerate(batch["name"]):
            tumor, view = parse_name(name)
            area_ratio, group = size_group(masks[i])
            for threshold in thresholds:
                dice, iou, precision, recall = sample_metrics(probs[i], masks[i], threshold)
                rows.append(
                    {
                        **candidate,
                        "threshold": threshold,
                        "name": name,
                        "tumor": tumor,
                        "view": view,
                        "area_ratio": area_ratio,
                        "size_group": group,
                        "dice": dice,
                        "iou": iou,
                        "precision": precision,
                        "recall": recall,
                    }
                )
    return rows


def write_dicts(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_by_threshold(rows: list[dict]) -> list[dict]:
    summary = []
    for candidate_name in sorted({row["candidate"] for row in rows}):
        candidate_rows = [row for row in rows if row["candidate"] == candidate_name]
        for threshold in sorted({row["threshold"] for row in candidate_rows}):
            threshold_rows = [row for row in candidate_rows if row["threshold"] == threshold]
            base = threshold_rows[0]
            summary.append(
                {
                    "candidate": candidate_name,
                    "checkpoint": base["checkpoint"],
                    "image_size": base["image_size"],
                    "boundary_weight": base["boundary_weight"],
                    "oversampling_weight": base["oversampling_weight"],
                    "loss_variant": base["loss_variant"],
                    "threshold": threshold,
                    **mean_rows(threshold_rows),
                }
            )
    return summary


def add_small_metrics(summary_rows: list[dict], rows: list[dict]) -> None:
    for summary in summary_rows:
        small_rows = [
            row
            for row in rows
            if row["candidate"] == summary["candidate"]
            and row["threshold"] == summary["threshold"]
            and row["size_group"] == "small_<1%"
        ]
        small = mean_rows(small_rows)
        summary["small_val_dice"] = small["val_dice"]
        summary["small_val_iou"] = small["val_iou"]
        summary["small_val_precision"] = small["val_precision"]
        summary["small_val_recall"] = small["val_recall"]


def group_metrics(rows: list[dict], metric_prefix: str) -> list[dict]:
    output = []
    for group_type in ["tumor", "view", "size_group"]:
        for group in sorted({row[group_type] for row in rows}):
            group_rows = [row for row in rows if row[group_type] == group]
            means = mean_test_rows(group_rows) if metric_prefix == "test" else mean_rows(group_rows)
            output.append({"group_type": group_type, "group": group, **means})
    return output


def save_visual_grid(rows: list[dict], out_path: Path, max_visuals: int) -> None:
    rows = sorted(rows, key=lambda row: row["dice"])
    selected = rows if len(rows) <= max_visuals else rows[: max_visuals // 2] + rows[-(max_visuals - max_visuals // 2) :]
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


def candidate_dict(raw: tuple) -> dict:
    name, checkpoint, image_size, boundary_weight, oversampling_weight, loss_variant = raw
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "candidate": name,
        "checkpoint": str(path),
        "image_size": image_size,
        "boundary_weight": boundary_weight,
        "oversampling_weight": oversampling_weight,
        "loss_variant": loss_variant,
    }


def run_val_eval(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print("split=val")

    all_rows = []
    for raw in CANDIDATES:
        candidate = candidate_dict(raw)
        _, dataset = make_train_val_datasets(
            args.data_root,
            image_size=candidate["image_size"],
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(f"evaluating {candidate['candidate']} image_size={candidate['image_size']} n={len(dataset)}")
        all_rows.extend(collect_rows(candidate, dataset, args.thresholds, device, args.batch_size, args.base_channels))

    sweep_rows = summarize_by_threshold(all_rows)
    add_small_metrics(sweep_rows, all_rows)
    summary_rows = []
    for candidate_name in sorted({row["candidate"] for row in sweep_rows}):
        candidate_sweeps = [row for row in sweep_rows if row["candidate"] == candidate_name]
        summary_rows.append(
            max(
                candidate_sweeps,
                key=lambda row: (row["val_dice"], row["val_iou"], row["val_precision"], row["val_recall"]),
            )
        )
    summary_rows = sorted(summary_rows, key=lambda row: row["val_dice"], reverse=True)
    selected = summary_rows[0]

    sweep_fields = [
        "candidate",
        "checkpoint",
        "image_size",
        "boundary_weight",
        "oversampling_weight",
        "loss_variant",
        "threshold",
        "n_samples",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
        "small_val_dice",
        "small_val_iou",
        "small_val_precision",
        "small_val_recall",
    ]
    write_dicts(args.out_dir / "threshold_sweep_val.csv", sweep_rows, sweep_fields)
    write_dicts(args.out_dir / "summary_val.csv", summary_rows, sweep_fields)

    selected_rows = [
        row for row in all_rows if row["candidate"] == selected["candidate"] and row["threshold"] == selected["threshold"]
    ]
    write_dicts(
        args.out_dir / "summary_val_group_metrics.csv",
        group_metrics(selected_rows, "val"),
        ["group_type", "group", "n_samples", "val_dice", "val_iou", "val_precision", "val_recall"],
    )
    with (args.out_dir / "final_selection.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_split": "val",
                "selection_metric": "per-sample mean val Dice",
                "tie_breakers": ["val_iou", "val_precision", "val_recall"],
                "selected": selected,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"selected={selected['candidate']} threshold={selected['threshold']}")
    print(f"val_dice={selected['val_dice']:.4f} val_iou={selected['val_iou']:.4f}")


@torch.no_grad()
def collect_test_rows(selection: dict, data_root: Path, device, batch_size: int, base_channels: int, max_visuals: int):
    dataset = BriscSegmentationDataset(data_root, split="test", image_size=int(selection["image_size"]), augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(base_channels).to(device)
    checkpoint = torch.load(selection["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    threshold = float(selection["threshold"])
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        probs = torch.sigmoid(model(images))
        preds = (probs > threshold).float()
        for i, name in enumerate(batch["name"]):
            tumor, view = parse_name(name)
            area_ratio, group = size_group(masks[i])
            dice, iou, precision, recall = sample_metrics(probs[i], masks[i], threshold)
            rows.append(
                {
                    "name": name,
                    "tumor": tumor,
                    "view": view,
                    "area_ratio": area_ratio,
                    "size_group": group,
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "image": images[i].detach().cpu(),
                    "mask": masks[i].detach().cpu(),
                    "prob": probs[i].detach().cpu(),
                    "pred": preds[i].detach().cpu(),
                }
            )
    return rows


def run_final_test(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.selection_json.open("r", encoding="utf-8") as f:
        selection = json.load(f)["selected"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"final_candidate={selection['candidate']} threshold={selection['threshold']}")

    rows = collect_test_rows(selection, args.data_root, device, args.batch_size, args.base_channels, args.max_visuals)
    metrics = mean_test_rows(rows)
    small = mean_test_rows([row for row in rows if row["size_group"] == "small_<1%"])
    final_row = {
        **selection,
        **metrics,
        "small_test_dice": small["test_dice"],
        "small_test_iou": small["test_iou"],
        "small_test_precision": small["test_precision"],
        "small_test_recall": small["test_recall"],
    }
    fields = [
        "candidate",
        "checkpoint",
        "image_size",
        "boundary_weight",
        "oversampling_weight",
        "loss_variant",
        "threshold",
        "n_samples",
        "test_dice",
        "test_iou",
        "test_precision",
        "test_recall",
        "small_test_dice",
        "small_test_iou",
        "small_test_precision",
        "small_test_recall",
    ]
    write_dicts(args.out_dir / "final_test.csv", [final_row], fields)
    write_dicts(
        args.out_dir / "final_test_group_metrics.csv",
        group_metrics(rows, "test"),
        ["group_type", "group", "n_samples", "test_dice", "test_iou", "test_precision", "test_recall"],
    )
    write_dicts(
        args.out_dir / "final_test_per_sample_metrics.csv",
        rows,
        ["name", "tumor", "view", "area_ratio", "size_group", "dice", "iou", "precision", "recall"],
    )
    save_visual_grid(rows, args.out_dir / "final_test_examples_worst_best.png", args.max_visuals)
    print(f"test_dice={metrics['test_dice']:.4f} test_iou={metrics['test_iou']:.4f}")


def main() -> None:
    args = parse_args()
    if args.command == "val-eval":
        run_val_eval(args)
    elif args.command == "final-test":
        run_final_test(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
