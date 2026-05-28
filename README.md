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
```

A3 / A4 / A5 后续继续补充。

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

### A1 与 A2 对比

| 实验 | 模型 | 验证集 Dice | 测试集 Dice |
|---|---|---:|---:|
| A1 | U-Net | 0.7938 | 0.7930 |
| A2 | Attention U-Net | 0.7851 | 0.7937 |

结论：

> A2 在测试集上相比 A1 略有提升，但幅度非常小，整体可以认为与 A1 基本持平。单独加入 Attention Gate 对整体 Dice / IoU 的提升有限，后续需要继续尝试 Boundary Loss 来改善边界和小目标分割。

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
│   ├── plot_history.py
│   ├── train_a1_unet.py
│   └── train_a2_attention_unet.py
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

### 6.6 独立测试集评估

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

### 6.7 生成分组可视化

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

后续路线：

```text
A3：U-Net + BCE + Dice + Boundary Loss
A4：Attention U-Net + BCE + Dice + Boundary Loss
A5：小 / 中 / 大肿瘤分组对比分析
```

重点观察：

- Boundary Loss 是否改善肿瘤边界；
- A4 是否优于 A1 / A2；
- 小肿瘤分割效果是否提升；
- 胶质瘤分割效果是否改善。
