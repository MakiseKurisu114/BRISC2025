from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history_csv: str | Path, out_path: str | Path, title: str) -> None:
    history_csv = Path(history_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss", linewidth=2)
    if "val_loss" in df:
        axes[0].plot(df["epoch"], df["val_loss"], label="val_loss", linewidth=2)
    if "eval_loss" in df:
        axes[0].plot(df["epoch"], df["eval_loss"], label="eval_loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_dice"], label="train_dice", linewidth=2)
    if "val_dice" in df:
        axes[1].plot(df["epoch"], df["val_dice"], label="val_dice", linewidth=2)
    if "eval_dice" in df:
        axes[1].plot(df["epoch"], df["eval_dice"], label="eval_dice", linewidth=2)
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(df["epoch"], df["train_iou"], label="train_iou", linewidth=2)
    if "val_iou" in df:
        axes[2].plot(df["epoch"], df["val_iou"], label="val_iou", linewidth=2)
    if "eval_iou" in df:
        axes[2].plot(df["epoch"], df["eval_iou"], label="eval_iou", linewidth=2)
    axes[2].set_title("IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
