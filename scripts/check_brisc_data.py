from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import BriscSegmentationDataset


def main() -> None:
    root = Path(".")
    for split in ["train", "test"]:
        dataset = BriscSegmentationDataset(root, split=split, image_size=128)
        print(f"{split}: {len(dataset)} paired image-mask samples")
        sample = dataset[0]
        print(
            f"  example={sample['name']} "
            f"image_shape={tuple(sample['image'].shape)} mask_shape={tuple(sample['mask'].shape)} "
            f"mask_values=({sample['mask'].min().item():.0f}, {sample['mask'].max().item():.0f})"
        )


if __name__ == "__main__":
    main()
