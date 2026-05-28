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

from src.dataset import make_train_val_datasets
from src.losses import BCEDiceBoundaryLoss
from src.metrics import dice_iou_from_logits
from src.model_unet import UNet
from src.plotting import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3: U-Net + BCE + Dice + Boundary Loss for BRISC 2025 segmentation")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/a3/full"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float, float]:
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        dice, iou = dice_iou_from_logits(logits.detach(), masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        n_batches += 1

    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


@torch.no_grad()
def save_prediction_figure(model, dataset, device, out_path: Path) -> None:
    model.eval()
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].squeeze().cpu()
    prob = torch.sigmoid(model(image)).squeeze().cpu()
    pred = (prob > 0.5).float()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(sample["image"].squeeze(), cmap="gray")
    axes[0].set_title("MRI")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("GT mask")
    axes[2].imshow(prob, cmap="gray")
    axes[2].set_title("Pred prob")
    axes[3].imshow(pred, cmap="gray")
    axes[3].set_title("Pred mask")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = make_train_val_datasets(
        args.data_root,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit=args.limit_train,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    criterion = BCEDiceBoundaryLoss(boundary_weight=args.boundary_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={device}")
    print(f"train_pairs={len(train_set)} val_pairs={len(val_set)} image_size={args.image_size}")
    print(f"boundary_weight={args.boundary_weight}")

    history_path = args.out_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou"])

        best_dice = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_dice, train_iou = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_dice, val_iou = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            writer.writerow([epoch, train_loss, train_dice, train_iou, val_loss, val_dice, val_iou])
            f.flush()

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_iou={train_iou:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
            )

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "val_dice": val_dice,
                    },
                    args.out_dir / "checkpoints" / "best_unet_boundary.pt",
                )

    save_prediction_figure(model, val_set, device, args.out_dir / "figures" / "sample_prediction.png")
    plot_training_history(history_path, args.out_dir / "figures" / "training_curves.png", "A3 U-Net + Boundary Loss")
    print(f"saved: {history_path}")
    print(f"saved: {args.out_dir / 'checkpoints' / 'best_unet_boundary.pt'}")
    print(f"saved: {args.out_dir / 'figures' / 'sample_prediction.png'}")
    print(f"saved: {args.out_dir / 'figures' / 'training_curves.png'}")


if __name__ == "__main__":
    main()
