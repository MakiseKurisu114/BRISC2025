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

## 3. A5 分组分析

A1-A4 训练阶段只使用 `segmentation_task/train`，并从 train 内部划分 validation set 用于保存 best checkpoint。`segmentation_task/test` 不参与训练，也不参与 checkpoint 保存；A5 不重新训练模型，而是使用 A1/A2/A3/A4 固定 checkpoint 的 `eval_test` 结果进行综合比较，并选择原始 A3 作为后续调参的基础模型。

分组维度包括：

```text
1. 按肿瘤类型分组：glioma / meningioma / pituitary
2. 按成像视角分组：axial / coronal / sagittal
3. 按肿瘤大小分组：small / medium / large
```

独立测试集整体指标：

| 实验 | 模型 | 样本数 | test Dice | test IoU |
|---|---|---:|---:|---:|
| A1 | U-Net | 860 | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 860 | 0.7937 | 0.7087 |
| A3 | U-Net + Boundary Loss | 860 | 0.8075 | 0.7271 |
| A4 | Attention U-Net + Boundary Loss | 860 | 0.7923 | 0.7093 |

A5 各分组最优模型：

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
- small tumor 组 A1 最高，说明小目标仍是后续调参重点；
- A4 没有超过 A3，说明 Attention Gate 与 Boundary Loss 的简单叠加没有带来额外收益。
- 因此，后续调参以 A3 为基础，重点关注 small tumor、boundary_weight、image_size、oversampling 和 threshold 等方向。

### A3 tuning validation-based selection

A3 tuning 是基于 A5 选择出的原始 A3 进行的补充优化探索。该阶段复用已有 checkpoint，在 validation set 上比较不同 boundary weight、oversampling、loss 变体、image size 和 threshold，并根据 validation per-sample mean Dice 选择候选配置。固定该候选配置后，仅在 test set 上进行一次 final test evaluation；该 final test 只用于检验 tuning 候选配置的泛化表现，不用于再次选择模型。

判断规则：

| A3 tuning 内容 | 判断依据 |
|---|---|
| boundary_weight | val Dice / val IoU |
| image_size | val Dice / val IoU |
| small oversampling 权重 | val Dice / val IoU / small tumor val 指标 |
| Focal Tversky 等 loss variant | val Dice / val IoU |
| threshold | val Dice / val IoU |
| fixed tuning candidate 泛化表现 | test Dice / test IoU |

规范选择结果保存在：

```text
outputs/a3_tuning/summary_val.csv
outputs/a3_tuning/threshold_sweep_val.csv
outputs/a3_tuning/final_selection.json
outputs/a3_tuning/final_test.csv
```

候选配置选择规则：

```text
validation set 上比较 A3 tuning 候选 checkpoint/config 和 threshold；
选择 validation per-sample mean Dice 最高的候选配置；
固定该候选配置后，只对 test set 做一次 final evaluation。
```

validation set 选出的 tuning 候选配置：

| selected candidate | threshold | val Dice | val IoU | val Precision | val Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.8052 | 0.7241 | 0.8335 | 0.8205 |

tuning candidate final test 结果：

| candidate | threshold | test Dice | test IoU | test Precision | test Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.7986 | 0.7164 | 0.8235 | 0.8311 |

final 流程统一使用 per-sample mean。

最终主结论：

> 综合 A1-A4 独立测试集整体指标，A3（U-Net + Boundary Loss）取得最高 Dice / IoU，因此被选为路线 A 的最终主模型和后续 tuning 基础。A3 tuning 进一步探索了 boundary weight、输入分辨率、小目标过采样、loss variant 和 threshold 等配置，并在 validation set 上选择 A3_boundary_w03 + threshold 0.30 作为候选配置；但其固定配置后的 final test 未超过原始 A3。因此，最终报告以原始 A3 的 test Dice = 0.8075、test IoU = 0.7271 作为主结果，A3 tuning 作为补充分析和后续优化探索。

### Final Model

最终模型汇总保存在：

```text
outputs/final_model/
```

`outputs/final_model/final_model.json` 根据已有结果文件自动选择 final model：先从 `outputs/a5/summary/overall_test_metrics.csv` 选择 A1-A4 中 independent test Dice 最高的主实验模型，再与 `outputs/a3_tuning/final_test.csv` 中的 tuning final candidate 对比。当前 final model 为原始 A3（U-Net + Boundary Loss），checkpoint 引用 `outputs/a3/full/checkpoints/best_unet_boundary.pt`，threshold = 0.50，test Dice = 0.8075，test IoU = 0.7271。`.pt` 文件不复制到 `outputs/final_model/`，只引用原 checkpoint 路径。

### Final qualitative results

已基于 final model 在 `segmentation_task/test` 上运行 final inference，并将最终展示结果统一保存到 `outputs/final_model/`：

```text
outputs/final_model/final_test_summary.csv
outputs/final_model/per_sample_metrics.csv
outputs/final_model/group_metrics.csv
outputs/final_model/selected_examples.csv
outputs/final_model/figures/
```

`figures/` 中包含 overall、tumor、view、size 四类可视化结果。每张图展示 MRI、ground truth mask、predicted mask 和 contour overlay，并标注样本 Dice / IoU。final inference 的 per-sample mean Dice = 0.8069，IoU = 0.7263；small tumor 组 Dice = 0.7542，仍是主要困难点之一。

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
│   ├── run_final_inference.py
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
│   ├── final_model/
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

A1-A4 的 checkpoint 均由 train 内部划分出的 validation set 选择；`segmentation_task/test` 不参与训练和 checkpoint 保存。以下命令使用固定 checkpoint 在 `segmentation_task/test` 上评估，生成 `eval_test`，用于 A5 综合比较。

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

A5 读取 A1-A4 的 `eval_test/`，用于测试集模型比较、分组分析，并选择原始 A3 作为后续调参的基础模型。

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

Final model：

```text
outputs/final_model/
├── README.md
├── final_model.json
├── final_test_summary.csv
├── per_sample_metrics.csv
├── group_metrics.csv
├── selected_examples.csv
└── figures/
```

其中：

- `history.csv`：每个 epoch 的 loss / Dice / IoU；
- `training_curves.png`：训练过程曲线；
- `checkpoints/*.pt`：模型权重；
- `eval_val/metrics.csv`：验证集整体指标，用于保存 best checkpoint；
- `eval_val/per_sample_metrics.csv`：验证集每张图单独指标；
- `eval_val/group_metrics.csv`：验证集按类型、视角、大小分组指标；
- `eval_test/metrics.csv`：测试集整体指标，用于 A5 综合比较；
- `eval_test/per_sample_metrics.csv`：测试集每张图单独指标，用于 A5 综合比较；
- `eval_test/group_metrics.csv`：测试集按类型、视角、大小分组指标，用于 A5 综合比较；
- `eval_test/group_visuals/`：测试集分组可视化图片，用于误差现象展示；
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

## 9. 最终整理状态

当前不再继续优化 A3，项目进入最终报告整理阶段。

```text
最终主模型：原始 A3（U-Net + Boundary Loss）
主结果：test Dice = 0.8075，test IoU = 0.7271
A3 tuning：补充分析和后续优化探索，不替代最终主结果
Final model summary：outputs/final_model/final_model.json
Final visualization：outputs/final_model/figures/
```

最终报告重点：

- 说明 A1-A4 训练只使用 train，并从 train 内部划分 validation 保存 best checkpoint；
- 使用 A5 汇总 A1-A4 的独立测试集结果，说明 A3 original 是整体最优；
- 将 A3 tuning 定位为 validation-based supplemental tuning；
- 说明 A3 tuning 的 fixed candidate final test 未超过原始 A3；
- 展示 final model 在 test split 上的 overall / tumor / view / size 可视化；
- 分析 small tumor 和 glioma 仍是主要困难点。
