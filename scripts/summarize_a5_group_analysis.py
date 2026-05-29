import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-brisc")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS = [
    ("A1", "U-Net", Path("outputs/a1/full/eval_val")),
    ("A2", "Attention U-Net", Path("outputs/a2/full/eval_val")),
    ("A3", "U-Net + Boundary Loss", Path("outputs/a3/full/eval_val")),
    ("A4", "Attention U-Net + Boundary Loss", Path("outputs/a4/full/eval_val")),
]

GROUP_ORDER = {
    "tumor": ["glioma", "meningioma", "pituitary"],
    "view": ["axial", "coronal", "sagittal"],
    "size_group": ["small_<1%", "medium_1-5%", "large_>5%"],
}

GROUP_LABELS = {
    "small_<1%": "small < 1%",
    "medium_1-5%": "medium 1%-5%",
    "large_>5%": "large > 5%",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A5-val: summarize A1-A4 validation group analysis")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/a5/summary"))
    return parser.parse_args()


def read_overall_metrics() -> pd.DataFrame:
    rows = []
    for exp_id, model, eval_dir in EXPERIMENTS:
        metrics = pd.read_csv(eval_dir / "metrics.csv").iloc[0]
        rows.append(
            {
                "experiment": exp_id,
                "model": model,
                "n_samples": int(metrics["n_samples"]),
                "val_dice": float(metrics["mean_dice"]),
                "val_iou": float(metrics["mean_iou"]),
            }
        )
    return pd.DataFrame(rows)


def read_group_metrics() -> pd.DataFrame:
    frames = []
    for exp_id, model, eval_dir in EXPERIMENTS:
        frame = pd.read_csv(eval_dir / "group_metrics.csv")
        frame.insert(0, "model", model)
        frame.insert(0, "experiment", exp_id)
        frames.append(frame)
    group_df = pd.concat(frames, ignore_index=True)
    group_df["group_label"] = group_df["group"].map(GROUP_LABELS).fillna(group_df["group"])
    return group_df


def best_by_group(group_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group_type, group), frame in group_df.groupby(["group_type", "group"], sort=False):
        best_dice = frame.sort_values("mean_dice", ascending=False).iloc[0]
        best_iou = frame.sort_values("mean_iou", ascending=False).iloc[0]
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "group_label": GROUP_LABELS.get(group, group),
                "n": int(best_dice["n"]),
                "best_dice_experiment": best_dice["experiment"],
                "best_dice_model": best_dice["model"],
                "best_dice": float(best_dice["mean_dice"]),
                "best_iou_experiment": best_iou["experiment"],
                "best_iou_model": best_iou["model"],
                "best_iou": float(best_iou["mean_iou"]),
            }
        )
    return pd.DataFrame(rows)


def write_pivot_tables(group_df: pd.DataFrame, out_dir: Path) -> None:
    for metric in ["mean_dice", "mean_iou"]:
        for group_type in ["tumor", "view", "size_group"]:
            subset = group_df[group_df["group_type"] == group_type].copy()
            subset["group"] = pd.Categorical(subset["group"], GROUP_ORDER[group_type], ordered=True)
            pivot = subset.pivot(index="group", columns="experiment", values=metric).sort_index()
            pivot.index = [GROUP_LABELS.get(str(index), str(index)) for index in pivot.index]
            pivot.to_csv(out_dir / f"{group_type}_{metric}_pivot.csv")


def plot_overall(overall_df: pd.DataFrame, out_path: Path) -> None:
    x = range(len(overall_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], overall_df["val_dice"], width=width, label="Dice")
    ax.bar([i + width / 2 for i in x], overall_df["val_iou"], width=width, label="IoU")
    ax.set_xticks(list(x))
    ax.set_xticklabels(overall_df["experiment"])
    ax.set_ylim(0.65, 0.84)
    ax.set_ylabel("Score")
    ax.set_title("A1-A4 Overall Validation Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for i, row in overall_df.iterrows():
        ax.text(i - width / 2, row["val_dice"] + 0.003, f"{row['val_dice']:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, row["val_iou"] + 0.003, f"{row['val_iou']:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_group_metric(group_df: pd.DataFrame, group_type: str, metric: str, out_path: Path) -> None:
    subset = group_df[group_df["group_type"] == group_type].copy()
    subset["group"] = pd.Categorical(subset["group"], GROUP_ORDER[group_type], ordered=True)
    subset = subset.sort_values(["group", "experiment"])
    pivot = subset.pivot(index="group", columns="experiment", values=metric).sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(0.48 if group_type == "tumor" else 0.62, 0.94 if group_type == "tumor" else 0.88)
    ax.set_xlabel("")
    ax.set_ylabel("Dice" if metric == "mean_dice" else "IoU")
    ax.set_title(f"A1-A4 {group_type.replace('_', ' ').title()} {ax.get_ylabel()} Comparison")
    ax.set_xticklabels([GROUP_LABELS.get(str(item), str(item)) for item in pivot.index], rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Experiment", ncols=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], float_columns: set[str] | None = None) -> str:
    float_columns = float_columns or set()
    rows = []
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows.extend([header, separator])
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in float_columns:
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_markdown_summary(overall_df: pd.DataFrame, best_df: pd.DataFrame, out_path: Path) -> None:
    best_overall = overall_df.sort_values("val_dice", ascending=False).iloc[0]
    best_columns = [
        "group_type",
        "group_label",
        "n",
        "best_dice_experiment",
        "best_dice",
        "best_iou_experiment",
        "best_iou",
    ]
    lines = [
        "# A5 分组综合分析摘要",
        "",
        "A5-val 汇总 A1-A4 已有验证集结果，不重新训练模型。",
        "",
        "A5-val 用于模型比较、分组分析和后续调参方向判断；test set 只用于最终固定配置后的泛化评估。",
        "",
        "## 验证集整体指标",
        "",
        markdown_table(overall_df, ["experiment", "model", "n_samples", "val_dice", "val_iou"], {"val_dice", "val_iou"}),
        "",
        "## 各分组最优模型",
        "",
        markdown_table(best_df[best_columns], best_columns, {"best_dice", "best_iou"}),
        "",
        "## 主要结论",
        "",
        f"- 验证集整体最优模型是 {best_overall['experiment']}（{best_overall['model']}），Dice = {best_overall['val_dice']:.4f}，IoU = {best_overall['val_iou']:.4f}。",
        "- glioma 仍然是最难分割的肿瘤类型。",
        "- small tumor 仍然比 medium / large tumor 更困难。",
        "- 当前控制变量对比中，Boundary Loss 是最明确有效的改进。",
        "- 当前超参数下，Attention U-Net + Boundary Loss 没有超过 U-Net + Boundary Loss。",
        "- 后续 A3 tuning 以 validation-based selection 选择配置和 threshold。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    overall_df = read_overall_metrics()
    group_df = read_group_metrics()
    best_df = best_by_group(group_df)

    overall_df.to_csv(args.out_dir / "overall_val_metrics.csv", index=False)
    group_df.to_csv(args.out_dir / "group_val_metrics_long.csv", index=False)
    best_df.to_csv(args.out_dir / "best_by_group_val.csv", index=False)
    write_pivot_tables(group_df, args.out_dir)

    plot_overall(overall_df, figures_dir / "overall_val_metrics.png")
    for group_type in ["tumor", "view", "size_group"]:
        plot_group_metric(group_df, group_type, "mean_dice", figures_dir / f"{group_type}_dice_comparison.png")
        plot_group_metric(group_df, group_type, "mean_iou", figures_dir / f"{group_type}_iou_comparison.png")

    write_markdown_summary(overall_df, best_df, args.out_dir / "README.md")

    print(f"saved: {args.out_dir / 'overall_val_metrics.csv'}")
    print(f"saved: {args.out_dir / 'group_val_metrics_long.csv'}")
    print(f"saved: {args.out_dir / 'best_by_group_val.csv'}")
    print(f"saved: {args.out_dir / 'README.md'}")
    print(f"saved figures: {figures_dir}")


if __name__ == "__main__":
    main()
