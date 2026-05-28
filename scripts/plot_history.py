import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-brisc")))

from src.plotting import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from history.csv")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_training_history(args.history, args.out, args.title)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
