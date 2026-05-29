# BRISC 2025 脑肿瘤 MRI 分割实验

本项目基于 Kaggle 数据集 **BRISC 2025**，完成医学图像处理课程期末项目中的脑肿瘤 MRI 图像分割任务。

项目选择路线 A：

```text
U-Net baseline
+ Attention U-Net
+ Boundary Loss
+ 小目标分组分析
```

目前已完成：

```text
A1：U-Net baseline
A2：Attention U-Net
A3：U-Net + Boundary Loss
A4：Attention U-Net + Boundary Loss
A5：A1-A4 分组综合分析
```

路线 A 主体实验已完成。

---

## 1. 项目任务

本项目目标是：

> 输入一张脑部 MRI 图像，模型自动预测肿瘤区域 mask。

使用的数据来自 BRISC 2025 的分割任务：

```text
segmentation_task/train/images
segmentation_task/train/masks
segmentation_task/test/images
segmentation_task/test/masks
```

数据规模：

```text
训练集：3933 对 image-mask
测试集：860 对 image-mask
```

说明：

- `images` 是 MRI 原图；
- `masks` 是对应的肿瘤标注；
- mask 中背景为 0，肿瘤区域为 1；
- 图像文件和 mask 文件通过同名 stem 一一对应。

---

## 2. 当前实验结果

### A1：U-Net Baseline

模型：

```text
U-Net + BCE Loss + Dice Loss
```

主要结果：

| 数据集 | Dice | IoU |
|---|---:|---:|
| 验证集 | 0.7938 | 0.7082 |
| 测试集 | 0.7930 | 0.7078 |

A1 小数据集过拟合测试：

```text
8 张图像
eval_dice = 0.9773
eval_iou  = 0.9565
```

说明 U-Net 模型、loss、image-mask 配对和训练流程可以正常学习。

### A2：Attention U-Net

模型：

```text
Attention U-Net + BCE Loss + Dice Loss
```

主要结果：

| 数据集 | Dice | IoU |
|---|---:|---:|
| 验证集 | 0.7851 | 0.7011 |
| 测试集 | 0.7937 | 0.7087 |

A2 小数据集过拟合测试：

```text
8 张图像
eval_dice = 0.9912
eval_iou  = 0.9828
```

### A3：U-Net + Boundary Loss

模型：

```text
U-Net + BCE Loss + Dice Loss + Boundary Loss
```

主要结果：

| 数据集 | Dice | IoU |
|---|---:|---:|
| 验证集 | 0.8043 | 0.7228 |
| 测试集 | 0.8075 | 0.7271 |

A3 小数据集过拟合测试：

```text
8 张图像
eval_dice = 0.9888
eval_iou  = 0.9779
```

### A4：Attention U-Net + Boundary Loss

模型：

```text
Attention U-Net + BCE Loss + Dice Loss + Boundary Loss
```

主要结果：

| 数据集 | Dice | IoU |
|---|---:|---:|
| 验证集 | 0.7934 | 0.7110 |
| 测试集 | 0.7923 | 0.7093 |

A4 小数据集过拟合测试：

```text
8 张图像
eval_dice = 0.9812
eval_iou  = 0.9634
```

### A1 / A2 / A3 / A4 对比

| 实验 | 模型 | 验证集 Dice | 验证集 IoU | 测试集 Dice | 测试集 IoU |
|---|---|---:|---:|---:|---:|
| A1 | U-Net | 0.7938 | 0.7082 | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 0.7851 | 0.7011 | 0.7937 | 0.7087 |
| A3 | U-Net + Boundary Loss | 0.8043 | 0.7228 | 0.8075 | 0.7271 |
| A4 | Attention U-Net + Boundary Loss | 0.7934 | 0.7110 | 0.7923 | 0.7093 |

结论：

> A2 在测试集上相比 A1 略有提升，但幅度非常小，整体可以认为与 A1 基本持平。A3 在验证集和独立测试集上均超过 A1/A2。A4 可以正常训练和过拟合小数据，但在当前设置下没有超过 A3，说明 Attention Gate 与 Boundary Loss 的简单叠加未带来额外整体收益。

---

## 3. 分组分析

项目已对测试集进行分组分析，包括：

```text
1. 按肿瘤类型分组：glioma / meningioma / pituitary
2. 按成像视角分组：axial / coronal / sagittal
3. 按肿瘤大小分组：small / medium / large
```

A1 测试集分组结果示例：

| 分组 | Dice | IoU |
|---|---:|---:|
| glioma | 0.6482 | 0.5446 |
| meningioma | 0.8998 | 0.8374 |
| pituitary | 0.8047 | 0.7113 |
| small < 1% | 0.7589 | 0.6649 |
| medium 1%-5% | 0.8142 | 0.7334 |
| large > 5% | 0.8190 | 0.7492 |

主要观察：

- 胶质瘤分割效果明显低于脑膜瘤和垂体瘤；
- 小肿瘤组效果低于中、大肿瘤组；
- 不同成像视角之间差距较小；
- 这些结果支持后续继续做 Boundary Loss 和小目标分析。

A3 测试集分组结果：

| 分组 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| glioma | 254 | 0.6619 | 0.5619 |
| meningioma | 306 | 0.9099 | 0.8552 |
| pituitary | 300 | 0.8245 | 0.7340 |
| axial | 346 | 0.8016 | 0.7252 |
| coronal | 257 | 0.8041 | 0.7193 |
| sagittal | 257 | 0.8168 | 0.7348 |
| small < 1% | 346 | 0.7542 | 0.6638 |
| medium 1%-5% | 454 | 0.8423 | 0.7667 |
| large > 5% | 60 | 0.8423 | 0.7809 |

A4 测试集分组结果：

| 分组 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| glioma | 254 | 0.6362 | 0.5376 |
| meningioma | 306 | 0.9088 | 0.8505 |
| pituitary | 300 | 0.8036 | 0.7082 |
| axial | 346 | 0.7959 | 0.7135 |
| coronal | 257 | 0.7987 | 0.7148 |
| sagittal | 257 | 0.7786 | 0.6953 |
| small < 1% | 346 | 0.7497 | 0.6519 |
| medium 1%-5% | 454 | 0.8218 | 0.7484 |
| large > 5% | 60 | 0.8039 | 0.7321 |

A5 综合分组结论：

| 分组 | 样本数 | Dice 最优 | 最优 Dice | IoU 最优 | 最优 IoU |
|---|---:|---|---:|---|---:|
| glioma | 254 | A3 | 0.6619 | A3 | 0.5619 |
| meningioma | 306 | A3 | 0.9099 | A3 | 0.8552 |
| pituitary | 300 | A2 | 0.8246 | A3 | 0.7340 |
| axial | 346 | A3 | 0.8016 | A3 | 0.7252 |
| coronal | 257 | A3 | 0.8041 | A3 | 0.7193 |
| sagittal | 257 | A3 | 0.8168 | A3 | 0.7348 |
| small < 1% | 346 | A1 | 0.7589 | A1 | 0.6649 |
| medium 1%-5% | 454 | A3 | 0.8423 | A3 | 0.7667 |
| large > 5% | 60 | A2 | 0.8516 | A3 | 0.7809 |

主要观察：

- A3 是整体测试集 Dice / IoU 最优模型；
- A3 在 glioma、meningioma、全部视角和 medium tumor 上表现最好；
- small tumor 组 A1 最高，说明 Boundary Loss 并没有在当前设置下稳定改善小目标；
- A4 没有超过 A3，说明 Attention Gate 与 Boundary Loss 的简单叠加没有带来额外收益。

### A3 tuning validation-based selection

A3 tuning 阶段复用已有 checkpoint，在 validation set 上比较不同 boundary weight、oversampling、loss 变体、image size 和 threshold，并根据 validation per-sample mean Dice 选择最终配置。固定最终配置后，仅在 test set 上进行一次最终评估。

规范选择结果保存在：

```text
outputs/a3_tuning/summary_val.csv
outputs/a3_tuning/threshold_sweep_val.csv
outputs/a3_tuning/final_selection.json
outputs/a3_tuning/final_test.csv
```

选择规则：

```text
validation set 上比较 A3 tuning 候选 checkpoint/config 和 threshold；
选择 validation per-sample mean Dice 最高的配置；
固定该配置后，只对 test set 做一次 final evaluation。
```

validation set 选出的最终配置：

| selected candidate | threshold | val Dice | val IoU | val Precision | val Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.8052 | 0.7241 | 0.8335 | 0.8205 |

final test 结果：

| candidate | threshold | test Dice | test IoU | test Precision | test Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.7986 | 0.7164 | 0.8235 | 0.8311 |

final 流程统一使用 per-sample mean。

---

## 4. 项目结构

```text
.
├── docs/
│   └── DATASET_README.md
├── scripts/
│   ├── check_brisc_data.py
│   ├── evaluate_checkpoint.py
│   ├── export_checkpoint_info.py
│   ├── generate_group_visuals.py
│   ├── overfit_a1_unet.py
│   ├── overfit_a2_attention_unet.py
│   ├── overfit_a3_unet_boundary.py
│   ├── overfit_a4_attention_unet_boundary.py
│   ├── plot_history.py
│   ├── summarize_a5_group_analysis.py
│   ├── train_a1_unet.py
│   ├── train_a2_attention_unet.py
│   ├── train_a3_unet_boundary.py
│   └── train_a4_attention_unet_boundary.py
├── src/
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model_attention_unet.py
│   ├── model_unet.py
│   └── plotting.py
├── outputs/
│   ├── a1/
│   ├── a2/
│   ├── a3/
│   ├── a4/
│   ├── a5/
│   └── README.md
├── 医学图像实验路线.md
└── 路线A实验记录.md
```

说明：

- `src/`：核心代码，包括数据集、模型、loss、metrics；
- `scripts/`：训练、评估、可视化和导出脚本；
- `outputs/`：实验结果；
- `路线A实验记录.md`：详细实验过程和结果记录；
- `docs/DATASET_README.md`：原始数据集说明。

---

## 5. 运行环境

本项目使用 Python + PyTorch。

当前本地环境：

```text
Python 3.12
PyTorch 2.7.1 + CUDA 11.8
GPU：NVIDIA GeForce RTX 3060 Laptop GPU
```

推荐使用项目外层已有虚拟环境：

```bash
/home/wxy/python_project/.venv/bin/python
```

---

## 6. 常用命令

### 6.1 检查数据集

```bash
/home/wxy/python_project/.venv/bin/python scripts/check_brisc_data.py
```

### 6.2 训练 A1：U-Net

```bash
/home/wxy/python_project/.venv/bin/python scripts/train_a1_unet.py \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --epochs 20 \
  --out-dir outputs/a1/full
```

### 6.3 训练 A2：Attention U-Net

```bash
/home/wxy/python_project/.venv/bin/python scripts/train_a2_attention_unet.py \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --epochs 20 \
  --out-dir outputs/a2/full
```

### 6.4 A1 小数据集过拟合测试

```bash
/home/wxy/python_project/.venv/bin/python scripts/overfit_a1_unet.py \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --num-samples 8 \
  --epochs 200 \
  --out-dir outputs/a1/overfit_8
```

### 6.5 A2 小数据集过拟合测试

```bash
/home/wxy/python_project/.venv/bin/python scripts/overfit_a2_attention_unet.py \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --num-samples 8 \
  --epochs 200 \
  --out-dir outputs/a2/overfit_8
```

### 6.6 训练 A3：U-Net + Boundary Loss

```bash
/home/wxy/python_project/.venv/bin/python scripts/train_a3_unet_boundary.py \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --epochs 20 \
  --boundary-weight 0.2 \
  --out-dir outputs/a3/full
```

### 6.7 A3 小数据集过拟合测试

```bash
/home/wxy/python_project/.venv/bin/python scripts/overfit_a3_unet_boundary.py \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --num-samples 8 \
  --epochs 200 \
  --boundary-weight 0.2 \
  --out-dir outputs/a3/overfit_8
```

### 6.8 训练 A4：Attention U-Net + Boundary Loss

```bash
/home/wxy/python_project/.venv/bin/python scripts/train_a4_attention_unet_boundary.py \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --epochs 20 \
  --boundary-weight 0.2 \
  --out-dir outputs/a4/full
```

### 6.9 A4 小数据集过拟合测试

```bash
/home/wxy/python_project/.venv/bin/python scripts/overfit_a4_attention_unet_boundary.py \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --num-samples 8 \
  --epochs 200 \
  --boundary-weight 0.2 \
  --out-dir outputs/a4/overfit_8
```

### 6.10 独立测试集评估

A1：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/a1/full/checkpoints/best_unet.pt \
  --model unet \
  --eval-split test \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --out-dir outputs/a1/full/eval_test
```

A2：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/a2/full/checkpoints/best_attention_unet.pt \
  --model attention_unet \
  --eval-split test \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --out-dir outputs/a2/full/eval_test
```

A3：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/a3/full/checkpoints/best_unet_boundary.pt \
  --model unet \
  --eval-split test \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --out-dir outputs/a3/full/eval_test
```

A4：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/a4/full/checkpoints/best_attention_unet_boundary.pt \
  --model attention_unet \
  --eval-split test \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --out-dir outputs/a4/full/eval_test
```

### 6.11 生成分组可视化

A1：

```bash
/home/wxy/python_project/.venv/bin/python scripts/generate_group_visuals.py \
  --checkpoint outputs/a1/full/checkpoints/best_unet.pt \
  --model unet \
  --eval-split test \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --out-dir outputs/a1/full/eval_test/group_visuals
```

A3：

```bash
/home/wxy/python_project/.venv/bin/python scripts/generate_group_visuals.py \
  --checkpoint outputs/a3/full/checkpoints/best_unet_boundary.pt \
  --model unet \
  --eval-split test \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --out-dir outputs/a3/full/eval_test/group_visuals
```

A4：

```bash
/home/wxy/python_project/.venv/bin/python scripts/generate_group_visuals.py \
  --checkpoint outputs/a4/full/checkpoints/best_attention_unet_boundary.pt \
  --model attention_unet \
  --eval-split test \
  --image-size 128 \
  --base-channels 16 \
  --batch-size 8 \
  --out-dir outputs/a4/full/eval_test/group_visuals
```

### 6.12 生成 A5 分组综合分析

```bash
/home/wxy/python_project/.venv/bin/python scripts/summarize_a5_group_analysis.py \
  --out-dir outputs/a5/summary
```

---

## 7. 输出结果说明

A1：

```text
outputs/a1/full/
├── checkpoints/best_unet.pt
├── history.csv
├── figures/training_curves.png
├── eval_val/
├── eval_test/
└── readable_checkpoint/
```

A2：

```text
outputs/a2/full/
├── checkpoints/best_attention_unet.pt
├── history.csv
├── figures/training_curves.png
├── eval_val/
├── eval_test/
└── readable_checkpoint/
```

A3：

```text
outputs/a3/full/
├── checkpoints/best_unet_boundary.pt
├── history.csv
├── figures/training_curves.png
├── eval_val/
├── eval_test/
└── readable_checkpoint/
```

A4：

```text
outputs/a4/full/
├── checkpoints/best_attention_unet_boundary.pt
├── history.csv
├── figures/training_curves.png
├── eval_val/
├── eval_test/
└── readable_checkpoint/
```

A5：

```text
outputs/a5/summary/
├── overall_test_metrics.csv
├── group_metrics_long.csv
├── best_by_group.csv
├── *_pivot.csv
├── figures/
└── README.md
```

其中：

- `history.csv`：每个 epoch 的 loss / Dice / IoU；
- `training_curves.png`：训练过程曲线；
- `checkpoints/*.pt`：模型权重；
- `eval_test/metrics.csv`：测试集整体指标；
- `eval_test/per_sample_metrics.csv`：每张图单独指标；
- `eval_test/group_metrics.csv`：按类型、视角、大小分组指标；
- `eval_test/group_visuals/`：分组可视化图片；
- `readable_checkpoint/`：可读模型结构和参数统计。

---

## 8. GitHub 注意事项

由于数据集和模型权重较大，建议不要上传：

```text
classification_task/
segmentation_task/
*.pt
outputs/**/checkpoints/
```

可以上传：

```text
src/
scripts/
docs/
outputs 中的 csv、png、README 等轻量结果
路线A实验记录.md
医学图像实验路线.md
README.md
```

---

## 9. 后续计划

后续可选工作：

```text
调参：A3 boundary_weight / A4 learning rate 与 epochs
```

重点观察：

- A3 的 boundary_weight 是否还能进一步提升；
- A4 是否需要更低学习率或更多 epoch；
- small tumor 组为什么 A1 最高；
- glioma 是否需要单独增强或类别针对性分析。
