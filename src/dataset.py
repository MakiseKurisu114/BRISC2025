from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BriscSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        limit: int | None = None,
        pairs: list[tuple[Path, Path]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        if pairs is not None:
            self.pairs = pairs[:limit] if limit is not None else pairs
            return

        image_dir = self.root / "segmentation_task" / split / "images"
        mask_dir = self.root / "segmentation_task" / split / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"Missing image/mask folder under {self.root}")
        image_paths = sorted(image_dir.glob("*.jpg"))
        pairs = []
        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                pairs.append((image_path, mask_path))

        if not pairs:
            raise RuntimeError(f"No paired image-mask files found in {image_dir}")

        self.pairs = pairs[:limit] if limit is not None else pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment:
            image_np, mask_np = self._augment(image_np, mask_np)

        image_tensor = torch.from_numpy(image_np.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": image_path.stem,
        }

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if np.random.rand() < 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        k = np.random.randint(0, 4)
        if k:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        return image, mask


def split_train_val(dataset: Dataset, val_ratio: float = 0.2, seed: int = 42):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)


def make_train_val_datasets(
    root: str | Path,
    image_size: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    limit: int | None = None,
) -> tuple[BriscSegmentationDataset, BriscSegmentationDataset]:
    base = BriscSegmentationDataset(root, split="train", image_size=image_size, augment=False, limit=limit)
    n_total = len(base.pairs)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train:])

    train_pairs = [pair for idx, pair in enumerate(base.pairs) if idx in train_indices]
    val_pairs = [pair for idx, pair in enumerate(base.pairs) if idx in val_indices]

    train_dataset = BriscSegmentationDataset(root, split="train", image_size=image_size, augment=True, pairs=train_pairs)
    val_dataset = BriscSegmentationDataset(root, split="train", image_size=image_size, augment=False, pairs=val_pairs)
    return train_dataset, val_dataset
