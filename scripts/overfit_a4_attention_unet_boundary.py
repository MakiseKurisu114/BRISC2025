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

from src.dataset import BriscSegmentationDataset
from src.losses import BCEDiceBoundaryLoss
from src.metrics import dice_iou_from_logits
from src.model_attention_unet import AttentionUNet
from src.plotting import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check: overfit A4 Attention U-Net + Boundary Loss on a tiny BRISC subset"
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/a4/overfit_8"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        dice, iou = dice_iou_from_logits(logits, masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        n_batches += 1
    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
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
def save_visual_grid(model, dataset, device, out_path: Path, max_samples: int = 8) -> None:
    model.eval()
    n = min(max_samples, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for idx in range(n):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].squeeze().cpu()
        logits = model(image)
        prob = torch.sigmoid(logits).squeeze().cpu()
        pred = (prob > 0.5).float()
        dice, _ = dice_iou_from_logits(logits.cpu(), sample["mask"].unsqueeze(0))

        axes[idx][0].imshow(sample["image"].squeeze(), cmap="gray")
        axes[idx][0].set_title(f"MRI\n{sample['name']}")
        axes[idx][1].imshow(mask, cmap="gray")
        axes[idx][1].set_title("GT mask")
        axes[idx][2].imshow(prob, cmap="gray")
        axes[idx][2].set_title("Pred prob")
        axes[idx][3].imshow(pred, cmap="gray")
        axes[idx][3].set_title(f"Pred mask\nDice={dice:.3f}")
        for ax in axes[idx]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BriscSegmentationDataset(
        args.data_root,
        split="train",
        image_size=args.image_size,
        augment=False,
        limit=args.num_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    criterion = BCEDiceBoundaryLoss(boundary_weight=args.boundary_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history_path = args.out_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_dice", "train_iou", "eval_loss", "eval_dice", "eval_iou"])

        print(f"device={device}")
        print(f"num_samples={len(dataset)} image_size={args.image_size} epochs={args.epochs}")
        print(f"boundary_weight={args.boundary_weight}")

        best_dice = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_dice, train_iou = train_one_epoch(model, loader, criterion, optimizer, device)
            eval_loss, eval_dice, eval_iou = evaluate(model, eval_loader, criterion, device)
            writer.writerow([epoch, train_loss, train_dice, train_iou, eval_loss, eval_dice, eval_iou])
            f.flush()

            if eval_dice > best_dice:
                best_dice = eval_dice
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "eval_dice": eval_dice,
                    },
                    args.out_dir / "best_overfit_attention_unet_boundary.pt",
                )

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"epoch={epoch:03d} "
                    f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_iou={train_iou:.4f} "
                    f"eval_loss={eval_loss:.4f} eval_dice={eval_dice:.4f} eval_iou={eval_iou:.4f}"
                )

    save_visual_grid(model, dataset, device, args.out_dir / "overfit_examples.png")
    plot_training_history(
        history_path,
        args.out_dir / "overfit_curves.png",
        "A4 Attention U-Net + Boundary Loss 8-sample Overfit Check",
    )
    print(f"saved: {history_path}")
    print(f"saved: {args.out_dir / 'best_overfit_attention_unet_boundary.pt'}")
    print(f"saved: {args.out_dir / 'overfit_examples.png'}")
    print(f"saved: {args.out_dir / 'overfit_curves.png'}")


if __name__ == "__main__":
    main()
