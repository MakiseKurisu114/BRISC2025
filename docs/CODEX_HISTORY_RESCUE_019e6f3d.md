# Codex 历史补救摘要：完成 A3 测试评估及后续实验

原 Codex 会话：

```text
thread_name = 完成 A3 测试评估
session_id  = 019e6f3d-1e67-7f11-a56d-812d4a6a148d
time        = 2026-05-28 23:39 到 2026-05-29 01:10 左右
raw_file    = /home/wxy/.codex/sessions/2026/05/28/rollout-2026-05-28T23-38-57-019e6f3d-1e67-7f11-a56d-812d4a6a148d.jsonl
size        = 约 1.7 MB
```

这份文档用于恢复“最新那条爆掉的上下文”。上一份恢复文档是：

```text
docs/CODEX_HISTORY_RESCUE_019e6e35.md
```

## 一句话结论

最新会话已经完成了 A3 的 test 评估、A4、A5 总结分析，以及围绕 small tumor 的 A3 调参实验。当前最好整体模型仍是 A3 original；small tumor 单项最好的是 A3 small oversampling w=3，但它牺牲了整体 Dice。

## 最新项目状态

项目路径：

```text
/home/wxy/python_project/kaggle/brisc2025
```

当前路线 A 进度：

```text
A1：完成
A2：完成
A3：完成，包括 test 评估和分组图
A4：完成
A5：完成，总结 A1/A2/A3/A4 的整体和分组对比
A3 tuning：完成 image-size、small oversampling、threshold sweep
```

当前 `git status --short` 只显示：

```text
?? docs/
```

说明主要代码、输出、报告大概率已经在之前提交过；`docs/` 是恢复文档目录，用户之前说过不需要上传 GitHub。

## A3 已补完

上一条历史中 A3 缺独立测试集评估和测试集分组图。最新会话已补完。

A3 模型：

```text
U-Net + BCE Loss + Dice Loss + Boundary Loss
boundary_weight = 0.2
image_size = 128
```

A3 test 结果：

| 模型 | test Dice | test IoU | 样本数 |
|---|---:|---:|---:|
| A3 U-Net + Boundary Loss | 0.8075 | 0.7271 | 860 |

主要输出：

```text
outputs/a3/full/eval_test/metrics.csv
outputs/a3/full/eval_test/group_metrics.csv
outputs/a3/full/eval_test/per_sample_metrics.csv
outputs/a3/full/eval_test/test_examples_worst_best.png
outputs/a3/full/eval_test/group_visuals/
```

结论：

- A3 在 test Dice / IoU 上超过 A1 和 A2。
- Boundary Loss 对当前任务是有效改进。

## A4 已完成

A4 模型：

```text
Attention U-Net + BCE Loss + Dice Loss + Boundary Loss
```

已完成内容：

- 全量训练。
- 小数据集 8 张图过拟合。
- val/test 评估。
- test 分组图。
- 可读 checkpoint。
- 写入 README、outputs/README、路线A实验记录。

A4 test 结果：

| 模型 | test Dice | test IoU | 样本数 |
|---|---:|---:|---:|
| A4 Attention U-Net + Boundary Loss | 0.7923 | 0.7093 | 860 |

A4 小数据过拟合：

```text
eval_dice = 0.9812
eval_iou  = 0.9634
```

主要输出：

```text
outputs/a4/full/
outputs/a4/full/eval_test/
outputs/a4/full/eval_test/group_visuals/
outputs/a4/full/readable_checkpoint/
outputs/a4/overfit_8/
```

结论：

- A4 能正常训练，也能小数据过拟合。
- A4 没有超过 A3。
- 注意力机制 + 边界损失并没有叠加出更好整体效果，可能是复杂模型更难调，或者当前数据/训练设置下 Attention Gate 没带来收益。

## A1/A2/A3/A4 最终 test 对比

| 实验 | 模型 | test Dice | test IoU | 样本数 |
|---|---|---:|---:|---:|
| A1 | U-Net | 0.7930 | 0.7078 | 860 |
| A2 | Attention U-Net | 0.7937 | 0.7087 | 860 |
| A3 | U-Net + Boundary Loss | 0.8075 | 0.7271 | 860 |
| A4 | Attention U-Net + Boundary Loss | 0.7923 | 0.7093 | 860 |

核心结论：

```text
A3 是当前整体最优模型。
```

## A5 已完成

A5 是总结分析，不是新模型。它把 A1/A2/A3/A4 在 test 集上的整体指标和分组指标统一汇总。

主要输出：

```text
outputs/a5/summary/README.md
outputs/a5/summary/overall_test_metrics.csv
outputs/a5/summary/group_metrics_long.csv
outputs/a5/summary/best_by_group.csv
outputs/a5/summary/figures/
```

A5 overall：

| 实验 | 模型 | test Dice | test IoU |
|---|---|---:|---:|
| A1 | U-Net | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 0.7937 | 0.7087 |
| A3 | U-Net + Boundary Loss | 0.8075 | 0.7271 |
| A4 | Attention U-Net + Boundary Loss | 0.7923 | 0.7093 |

分组最佳结论：

| 分组 | 最佳 Dice 模型 | Dice | 最佳 IoU 模型 | IoU |
|---|---|---:|---|---:|
| glioma | A3 | 0.6619 | A3 | 0.5619 |
| meningioma | A3 | 0.9099 | A3 | 0.8552 |
| pituitary | A2 | 0.8246 | A3 | 0.7340 |
| axial | A3 | 0.8016 | A3 | 0.7252 |
| coronal | A3 | 0.8041 | A3 | 0.7193 |
| sagittal | A3 | 0.8168 | A3 | 0.7348 |
| large > 5% | A2 | 0.8516 | A3 | 0.7809 |
| medium 1%-5% | A3 | 0.8423 | A3 | 0.7667 |
| small < 1% | A1 | 0.7589 | A1 | 0.6649 |

重要发现：

- A3 是总体最优。
- A3 对中等大小肿瘤、glioma、meningioma、不同视角整体更稳。
- small tumor 是明显短板，A3 overall 更好，但 small group 不如 A1。

## 后续调参方向讨论

用户发现 small tumor 难以发现，提出可能是 `image_size=128` 太小，小 tumor 像素太少，希望保住细节。

讨论后的调参优先级：

```text
1. 先试更大 image_size：192 / 256
2. 再试 small tumor oversampling
3. 再做 threshold sweep
4. 暂不优先引入 Focal Tversky，先用更可解释的改动
```

中间有一次用户要求撤销刚刚做的内容，随后按新的 small tumor 调参顺序继续做。

## A3 Image Size 调参

调参基线：

```text
A3 original: image_size = 128
```

尝试：

```text
image_size = 192
image_size = 256, batch_size = 8
```

结果：

| 实验 | image_size | test Dice | test IoU | small Dice | small IoU |
|---|---:|---:|---:|---:|---:|
| A3 original | 128 | 0.8075 | 0.7271 | 0.7542 | 0.6638 |
| A3 image_size 192 | 192 | 0.7809 | 0.6989 | 0.7447 | 0.6555 |
| A3 image_size 256 | 256 | 0.7744 | 0.6893 | 0.7414 | 0.6521 |

结论：

- 单纯加大输入分辨率没有改善 small tumor。
- 192/256 都让整体指标和 small 指标下降。
- 可能原因是训练轮数、batch、学习率、显存约束或模型容量还没配套调整。

输出目录：

```text
outputs/a3_tuning/image_size_192/full/
outputs/a3_tuning/image_size_256_bs8/full/
```

## A3 Small Tumor Oversampling

目的：

```text
让模型训练时更频繁看到 small tumor 样本。
```

尝试权重：

```text
small_sample_weight = 1.5
small_sample_weight = 2.0
small_sample_weight = 3.0
```

结果：

| 实验 | test Dice | test IoU | small Dice | small IoU |
|---|---:|---:|---:|---:|
| A3 original | 0.8075 | 0.7271 | 0.7542 | 0.6638 |
| oversampling w=1.5 | 0.7826 | 0.7020 | 0.7575 | 0.6699 |
| oversampling w=2.0 | 0.7974 | 0.7153 | 0.7487 | 0.6558 |
| oversampling w=3.0 | 0.7886 | 0.7024 | 0.7811 | 0.6890 |

结论：

- overall 最好：A3 original。
- small tumor 最好：oversampling w=3.0。
- oversampling 里整体折中最好：w=2.0，但它没有提升 small。
- w=1.5 只轻微提升 small，overall 掉得不划算。
- oversampling 能把模型往 small tumor 偏，但单靠采样权重很难同时提升 overall 和 small。

输出目录：

```text
outputs/a3_tuning/small_oversampling_w15/full/
outputs/a3_tuning/small_oversampling_w2/full/
outputs/a3_tuning/small_oversampling_w3/full/
outputs/a3_tuning/summary.csv
outputs/a3_tuning/README.md
```

## Threshold Sweep

用户问：

```text
先做阈值从 0.3 到 0.5 扫描，减少小目标被阈值直接抹掉？
```

于是对两个 checkpoint 做了阈值扫描：

```text
A3 original
A3 small oversampling w=3
```

扫描阈值：

```text
0.30 / 0.35 / 0.40 / 0.45 / 0.50
```

A3 original：

| threshold | overall Dice | overall IoU | small Dice | small IoU |
|---:|---:|---:|---:|---:|
| 0.30 | 0.8060 | 0.7251 | 0.7506 | 0.6593 |
| 0.35 | 0.8064 | 0.7256 | 0.7517 | 0.6607 |
| 0.40 | 0.8067 | 0.7261 | 0.7530 | 0.6624 |
| 0.45 | 0.8069 | 0.7263 | 0.7536 | 0.6630 |
| 0.50 | 0.8069 | 0.7263 | 0.7542 | 0.6638 |

A3 small oversampling w=3：

| threshold | overall Dice | overall IoU | small Dice | small IoU |
|---:|---:|---:|---:|---:|
| 0.30 | 0.7877 | 0.7006 | 0.7776 | 0.6838 |
| 0.35 | 0.7878 | 0.7010 | 0.7786 | 0.6854 |
| 0.40 | 0.7881 | 0.7014 | 0.7797 | 0.6867 |
| 0.45 | 0.7882 | 0.7018 | 0.7805 | 0.6880 |
| 0.50 | 0.7881 | 0.7018 | 0.7811 | 0.6890 |

结论：

- 降低阈值到 0.30-0.45 没有改善 small tumor。
- 默认阈值 0.50 反而是 small tumor 指标最高。
- small tumor 难点不主要是后处理阈值过高，更可能是定位不准、置信度分布不集中、采样和 loss 对小目标仍不够平衡。

输出目录：

```text
outputs/a3_tuning/threshold_sweep/
```

## 最新新增/关键脚本

```text
scripts/train_a4_attention_unet_boundary.py
scripts/overfit_a4_attention_unet_boundary.py
scripts/summarize_a5_group_analysis.py
scripts/train_a3_unet_boundary_small_oversampling.py
scripts/sweep_thresholds.py
```

核心已有脚本：

```text
scripts/train_a1_unet.py
scripts/train_a2_attention_unet.py
scripts/train_a3_unet_boundary.py
scripts/evaluate_checkpoint.py
scripts/generate_group_visuals.py
scripts/export_checkpoint_info.py
```

## 最新文档状态

最新会话中已更新：

```text
README.md
outputs/README.md
路线A实验记录.md
outputs/a3_tuning/README.md
outputs/a3_tuning/threshold_sweep/README.md
outputs/a5/summary/README.md
```

## 下一步建议

如果继续做实验，建议不要再盲目扩分辨率或降阈值。

更合理的下一步：

```text
以 A3 original 为主线，围绕 small tumor 改 loss 或训练目标。
```

可选方向：

```text
1. Dice + BCE + Focal Tversky，惩罚小病灶漏检。
2. 对 small tumor 单独调 loss 权重。
3. 以 A3 original 做主模型，A3 oversampling w=3 作为 small tumor 专项对照。
4. 报告中明确写：overall 和 small tumor 存在 tradeoff。
```

当前最适合写入报告的结论：

```text
A3 original 是综合最优模型；small tumor 是主要失败场景。Oversampling w=3 能显著提升 small tumor，但会牺牲整体性能。阈值扫描说明 small tumor 不是简单的后处理阈值问题。
```

## GitHub 提交命令

用户最后问“给我提交到 github 的指令”。当时给出的命令是：

```bash
git add README.md outputs src scripts 路线A实验记录.md
git commit -m "Add A3 small tumor tuning experiments"
git push origin main
```

注意：

```text
不要加 docs/
```

因为用户之前明确说 docs 文件夹不需要上传。当前 `docs/` 主要是本地历史恢复文档。

## 如果继续开新 Codex，可以复制这段

```text
这是 BRISC 2025 脑肿瘤 MRI 分割项目，路径是 /home/wxy/python_project/kaggle/brisc2025。请先读 docs/CODEX_HISTORY_RESCUE_019e6e35.md 和 docs/CODEX_HISTORY_RESCUE_019e6f3d.md。

当前路线 A 已完成 A1/A2/A3/A4/A5。A3 original 是整体最优：test Dice 0.8075, test IoU 0.7271。A5 已汇总 A1-A4 分组对比。small tumor 是主要短板，A1 在 small group 上最好，A3 overall 最好。

已做 A3 tuning：image_size 192/256 没提升；small oversampling w=3 提升 small Dice 到 0.7811 但整体下降；threshold sweep 0.30-0.50 没改善 small，默认 0.50 最好。

不要重跑 A1/A2/A3/A4/A5，除非明确要求。不要上传 docs/。如果继续实验，优先考虑 A3 original 基础上的 loss 改进，比如 Dice+BCE+Focal Tversky，并保留所有新结果到 outputs/a3_tuning/ 下，不覆盖旧结果。
```

