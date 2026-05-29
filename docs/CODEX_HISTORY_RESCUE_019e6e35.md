# Codex 历史补救摘要：分析项目是否用 Python

原 Codex 会话：

```text
thread_name = 分析项目是否用 Python
session_id  = 019e6e35-87f4-71d3-a00d-051acf3a3cdd
time        = 2026-05-28 18:51-21:08 左右
raw_file    = /home/wxy/.codex/sessions/2026/05/28/rollout-2026-05-28T18-51-02-019e6e35-87f4-71d3-a00d-051acf3a3cdd.jsonl
size        = 约 6 MB
```

这条历史打不开，大概率是因为会话太长，里面有大量工具调用、训练日志、补丁内容和 token 统计。原始文件没有损坏，重要可见内容已经压缩到本文档。

## 一句话结论

这个项目是 Python / PyTorch 项目，任务选择了 BRISC 2025 的医学图像分割方向。路线 A 已经完成 A1 和 A2，A3 已完成训练、验证集评估、小数据过拟合和验证集分组分析，但 A3 的独立测试集评估和测试集分组图还没跑完。

## 项目和数据集判断

- 项目目录：`/home/wxy/python_project/kaggle/brisc2025`
- 主要任务：BRISC 2025 脑肿瘤 MRI 分割。
- 语言和框架：Python，核心依赖 PyTorch。
- 数据集已由组员预处理好，已有训练、验证、测试划分，不需要额外重新切分。
- 训练时使用 `/home/wxy/python_project/.venv/bin/python`。
- 用户明确要求后续都优先使用 GPU。

## 路线 A 做什么

路线 A 是分割模型对比路线：

```text
A1：U-Net baseline
A2：Attention U-Net
A3：U-Net + Boundary Loss
A4：Attention U-Net + Boundary Loss
A5：按肿瘤类型、视角、大小做分组分析
```

已实际推进到 A3。A4/A5 后续可继续补。

## A1 已完成内容

模型：

```text
U-Net + BCE Loss + Dice Loss
```

完成内容：

- 全量训练。
- 验证集评估。
- 独立测试集评估。
- 小数据集 8 张图过拟合 sanity check。
- 训练曲线图。
- worst/best 预测可视化图。
- 按肿瘤类型、视角、肿瘤大小的分组指标与图片。
- 可读 checkpoint 摘要，避免只能看 `.pt` 二进制。
- 文件整理到 `outputs/a1/`。

核心结果：

| split | Dice | IoU | 样本数 |
|---|---:|---:|---:|
| val | 0.7938 | 0.7082 | 786 |
| test | 0.7930 | 0.7078 | 860 |

小数据过拟合结果：

```text
eval_dice = 0.9773
eval_iou  = 0.9565
```

主要文件：

```text
outputs/a1/full/history.csv
outputs/a1/full/figures/training_curves.png
outputs/a1/full/eval_val/
outputs/a1/full/eval_test/
outputs/a1/full/readable_checkpoint/
outputs/a1/overfit_8/
```

## A2 已完成内容

模型：

```text
Attention U-Net + BCE Loss + Dice Loss
```

完成内容与 A1 对齐：

- 全量训练。
- 验证集评估。
- 独立测试集评估。
- 小数据集 8 张图过拟合。
- 训练曲线图。
- worst/best 预测图。
- 分组指标与分组图。
- 可读 checkpoint 摘要。
- 文件整理到 `outputs/a2/`。

核心结果：

| split | Dice | IoU | 样本数 |
|---|---:|---:|---:|
| val | 0.7851 | 0.7011 | 786 |
| test | 0.7937 | 0.7087 | 860 |

小数据过拟合结果：

```text
eval_dice = 0.9912
eval_iou  = 0.9828
```

和 A1 对比：

- A2 加入 Attention Gate，理论上让模型更关注肿瘤区域。
- 当前实验中 A2 验证集略低于 A1。
- A2 测试集略高于 A1，但提升非常小，可以认为整体基本持平。
- 这属于合理结果：结构更复杂不必然带来明显提升，尤其数据量、训练轮数、超参数和肿瘤形态都会影响结果。

主要文件：

```text
outputs/a2/full/history.csv
outputs/a2/full/figures/training_curves.png
outputs/a2/full/eval_val/
outputs/a2/full/eval_test/
outputs/a2/full/readable_checkpoint/
outputs/a2/overfit_8/
```

## A3 已完成内容

模型：

```text
U-Net + BCE Loss + Dice Loss + Boundary Loss
boundary_weight = 0.2
```

A3 相比 A1/A2 多了 Boundary Loss。它不主要改网络结构，而是在损失函数里加入边界约束，让模型更关注肿瘤轮廓。

已完成：

- `src/losses.py` 增加 BoundaryLoss / BCEDiceBoundaryLoss。
- 新增 `scripts/train_a3_unet_boundary.py`。
- 新增 `scripts/overfit_a3_unet_boundary.py`。
- 完成 A3 全量训练。
- 完成 A3 验证集评估。
- 完成 A3 小数据过拟合。
- 完成 A3 验证集分组分析。
- 生成训练曲线、预测图、可读 checkpoint。
- 更新 `README.md`、`outputs/README.md`、`路线A实验记录.md`。

核心结果：

| split | Dice | IoU | 样本数 |
|---|---:|---:|---:|
| val | 0.8043 | 0.7228 | 786 |
| test | 待补 | 待补 | 待补 |

训练中最好结果：

```text
best_epoch = 18
val_dice   = 0.8043
val_iou    = 0.7228
```

小数据过拟合结果：

```text
eval_dice = 0.9888
eval_iou  = 0.9779
```

验证集分组结果：

按肿瘤类型：

| 类型 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| glioma | 231 | 0.6439 | 0.5398 |
| meningioma | 259 | 0.9114 | 0.8588 |
| pituitary | 296 | 0.8331 | 0.7436 |

按视角：

| 视角 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| axial | 251 | 0.7864 | 0.7064 |
| coronal | 271 | 0.7982 | 0.7078 |
| sagittal | 264 | 0.8246 | 0.7503 |

按肿瘤大小：

| 大小 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| small < 1% | 306 | 0.7549 | 0.6668 |
| medium 1%-5% | 458 | 0.8377 | 0.7604 |
| large > 5% | 22 | 0.7599 | 0.6778 |

主要文件：

```text
outputs/a3/full/history.csv
outputs/a3/full/figures/training_curves.png
outputs/a3/full/figures/sample_prediction.png
outputs/a3/full/eval_val/
outputs/a3/full/readable_checkpoint/
outputs/a3/full/checkpoints/best_unet_boundary.pt
outputs/a3/overfit_8/
```

## A1/A2/A3 当前对比

| 实验 | 模型 | val Dice | val IoU | test Dice | test IoU |
|---|---|---:|---:|---:|---:|
| A1 | U-Net | 0.7938 | 0.7082 | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 0.7851 | 0.7011 | 0.7937 | 0.7087 |
| A3 | U-Net + Boundary Loss | 0.8043 | 0.7228 | 待补 | 待补 |

目前能说的结论：

- A3 在验证集上最好，说明 Boundary Loss 对当前验证集有帮助。
- A2 相比 A1 没有明显优势，只在测试集上微小提升。
- A3 还不能和 A1/A2 做最终测试集结论，因为 A3 test 还没跑。

## 还没完成的关键任务

A3 差两步：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint outputs/a3/full/checkpoints/best_unet_boundary.pt --model unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a3/full/eval_test
```

```bash
/home/wxy/python_project/.venv/bin/python scripts/generate_group_visuals.py --checkpoint outputs/a3/full/checkpoints/best_unet_boundary.pt --model unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a3/full/eval_test/group_visuals
```

当时没有继续跑完的原因：GPU 执行被系统用量限制拦截，不是代码逻辑问题。

## Git 和文件管理状态

当前工作区有 A3 相关新改动：

```text
M  README.md
M  outputs/README.md
M  src/losses.py
M  路线A实验记录.md
?? outputs/a3/
?? scripts/overfit_a3_unet_boundary.py
?? scripts/train_a3_unet_boundary.py
```

`.pt` 权重文件应该被 `.gitignore` 忽略，不建议上传 GitHub。可读结果、CSV、图片和 Markdown 报告适合上传。

## 如果继续开新 Codex，可以复制这段

```text
这是 BRISC 2025 脑肿瘤 MRI 分割项目，路径是 /home/wxy/python_project/kaggle/brisc2025。历史里路线 A 已完成 A1/A2，并推进到 A3。请先阅读 docs/CODEX_HISTORY_RESCUE_019e6e35.md、README.md、路线A实验记录.md、outputs/README.md。

当前重点：继续完成 A3 的独立测试集评估和测试集分组可视化。使用 /home/wxy/python_project/.venv/bin/python 和 GPU。不要重做 A1/A2，不要删除已有结果。

A3 已有 checkpoint：
outputs/a3/full/checkpoints/best_unet_boundary.pt

需要运行：
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint outputs/a3/full/checkpoints/best_unet_boundary.pt --model unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a3/full/eval_test

/home/wxy/python_project/.venv/bin/python scripts/generate_group_visuals.py --checkpoint outputs/a3/full/checkpoints/best_unet_boundary.pt --model unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a3/full/eval_test/group_visuals

跑完后更新 README.md、outputs/README.md、路线A实验记录.md，并补充 A1/A2/A3 的最终 test 对比。
```

