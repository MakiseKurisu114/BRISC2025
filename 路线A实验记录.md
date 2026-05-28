# 路线 A 实验记录

项目任务：BRISC 2025 脑肿瘤 MRI 图像分割

路线 A：

```text
U-Net baseline
+ Attention U-Net
+ Boundary Loss
+ 小目标分组分析
```

本文件用于记录路线 A 每一步已经完成的内容、运行命令、实验结果和阶段结论。

---

## A1：U-Net Baseline

### 1. 实验目的

A1 的目标是先建立一个基础分割模型，作为后续 Attention U-Net、Boundary Loss 等改进方法的对照基准。

本步骤主要验证：

- BRISC 2025 分割数据能否正常读取；
- MRI 图像和 mask 是否能够正确配对；
- mask 是否能转换为二值标签；
- U-Net 模型是否能完成训练、验证和保存；
- Dice、IoU 等分割指标是否能正常计算。

简单来说，A1 是整条路线的 baseline，用来回答：

> 不加入注意力机制和边界损失时，普通 U-Net 在该数据集上的分割效果如何？

---

### 2. 使用数据

本实验使用 BRISC 2025 的分割任务数据：

```text
segmentation_task/train/images
segmentation_task/train/masks
```

数据检查结果：

```text
train: 3933 paired image-mask samples
test: 860 paired image-mask samples
mask_values=(0, 1)
```

说明：

- 训练集共有 3933 对 image-mask；
- 测试集共有 860 对 image-mask；
- mask 已经可以正确读取为二值图，背景为 0，肿瘤区域为 1。

---

### 3. 数据预处理

A1 中完成了以下代码层面的预处理：

- 读取灰度 MRI 图像；
- 读取对应的肿瘤 mask；
- 将图像统一缩放到 `128 x 128`；
- 将 MRI 图像像素归一化到 `[0, 1]`；
- 将 mask 二值化为 `0/1`；
- 对训练数据进行简单数据增强：
  - 随机水平翻转；
  - 随机垂直翻转；
  - 随机旋转 0/90/180/270 度；
- 从训练集中划分验证集。

实际划分结果：

```text
train_pairs = 3147
val_pairs   = 786
```

---

### 4. 模型方法

A1 使用基础 U-Net 作为 baseline。

模型结构：

```text
Input MRI image
   ↓
Encoder
   ↓
Bottleneck
   ↓
Decoder
   ↓
Predicted tumor mask
```

输入：

```text
1 通道 MRI 灰度图
```

输出：

```text
1 通道肿瘤区域预测 mask
```

损失函数：

```text
BCE Loss + Dice Loss
```

评价指标：

```text
Dice
IoU
```

---

### 5. 运行命令

正式 A1 baseline 使用 GPU 运行，命令如下：

```bash
/home/wxy/python_project/.venv/bin/python scripts/train_a1_unet.py --image-size 128 --base-channels 16 --batch-size 8 --epochs 20 --out-dir outputs/a1/full
```

运行设备：

```text
device = cuda
GPU = NVIDIA GeForce RTX 3060 Laptop GPU
```

---

### 6. 实验结果

训练设置：

```text
model         = U-Net
loss          = BCE Loss + Dice Loss
image_size    = 128
base_channels = 16
batch_size    = 8
epochs        = 20
train_pairs   = 3147
val_pairs     = 786
device        = cuda
```

最好验证集结果出现在第 19 个 epoch：

```text
best_epoch = 19
val_loss   = 0.1223
val_dice   = 0.7921
val_iou    = 0.7079
```

最后一轮结果：

```text
epoch      = 20
train_dice = 0.7985
train_iou  = 0.7122
val_dice   = 0.7802
val_iou    = 0.6983
```

验证集 Dice 变化趋势：

```text
epoch 1  : val_dice = 0.4309
epoch 5  : val_dice = 0.6737
epoch 10 : val_dice = 0.7681
epoch 15 : val_dice = 0.7899
epoch 19 : val_dice = 0.7921
epoch 20 : val_dice = 0.7802
```

---

### 7. 输出文件

A1 正式实验结果保存在：

```text
outputs/a1/full/
```

主要文件：

```text
outputs/a1/full/history.csv
outputs/a1/full/checkpoints/best_unet.pt
outputs/a1/full/figures/training_curves.png
outputs/a1/full/eval_val/val_examples_worst_best.png
outputs/a1/full/eval_val/group_metrics.csv
outputs/a1/full/eval_test/test_examples_worst_best.png
outputs/a1/full/eval_test/group_metrics.csv
```

文件说明：

- `history.csv`：记录每个 epoch 的 loss、Dice、IoU；
- `best_unet.pt`：验证集 Dice 最好的模型权重；
- `training_curves.png`：训练集和验证集的 loss、Dice、IoU 变化曲线；
- `eval_val/val_examples_worst_best.png`：验证集中最差和最好样本的可视化对比；
- `eval_test/test_examples_worst_best.png`：独立测试集中最差和最好样本的可视化对比；
- `group_metrics.csv`：按肿瘤类型、成像视角和肿瘤大小分组后的 Dice / IoU。

---

### 8. 阶段结论

A1 已经成功跑通，普通 U-Net 在 BRISC 2025 脑肿瘤 MRI 分割任务上取得了较好的 baseline 效果：

```text
best val_dice = 0.7921
best val_iou  = 0.7079
```

该结果说明：

- 基础 U-Net 已经能够学习到脑肿瘤区域的主要形态；
- 整体分割效果较稳定；
- 该结果可以作为后续 A2、A3、A4 的对照基准；
- 后续改进需要观察 Attention U-Net 和 Boundary Loss 是否能进一步提升 Dice、IoU，尤其是边界区域和小肿瘤样本上的表现。

---

### 9. 小数据集过拟合测试

为了验证 A1 的模型结构、损失函数、image-mask 配对和训练流程是否正确，额外进行了小数据集过拟合 sanity check。

实验设置：

```text
model       = U-Net
num_samples = 8
image_size  = 128
batch_size  = 8
epochs      = 200
augment     = False
device      = cuda
```

运行命令：

```bash
/home/wxy/python_project/.venv/bin/python scripts/overfit_a1_unet.py --image-size 128 --base-channels 16 --batch-size 8 --num-samples 8 --epochs 200 --out-dir outputs/a1/overfit_8
```

最终结果：

```text
epoch      = 200
train_loss = 0.3807
train_dice = 0.9635
train_iou  = 0.9311
eval_loss  = 0.3827
eval_dice  = 0.9773
eval_iou   = 0.9565
```

输出文件：

```text
outputs/a1/overfit_8/history.csv
outputs/a1/overfit_8/best_overfit_unet.pt
outputs/a1/overfit_8/overfit_examples.png
outputs/a1/overfit_8/overfit_curves.png
```

结论：

```text
U-Net 可以在 8 张训练图像上过拟合到 Dice ≈ 0.98。
```

这说明 A1 的基础训练流程是有效的，模型能够学习到给定 image-mask 的对应关系；如果正式验证集中仍然出现错分样本，更可能是泛化能力、样本难度、肿瘤大小或边界模糊导致，而不是代码完全无法学习。

---

### 10. 独立测试集评估与分组分析

为了观察 A1 baseline 在独立测试集上的泛化效果，使用训练好的 `best_unet.pt` 在 `segmentation_task/test` 上进行了单独评估。

运行命令：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint outputs/a1/full/checkpoints/best_unet.pt --model unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a1/full/eval_test
```

测试集整体结果：

```text
n_samples = 860
test_dice = 0.7930
test_iou  = 0.7078
```

验证集重新评估结果：

```text
n_samples = 786
val_dice  = 0.7938
val_iou   = 0.7082
```

验证集和测试集结果非常接近，说明 A1 baseline 没有明显只适配验证集，具备一定泛化能力。

#### 10.1 按肿瘤类型分组

测试集结果：

| 肿瘤类型 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| glioma | 254 | 0.6482 | 0.5446 |
| meningioma | 306 | 0.8998 | 0.8374 |
| pituitary | 300 | 0.8047 | 0.7113 |

可以看到，A1 对脑膜瘤分割效果最好，对胶质瘤分割效果明显较差。这说明普通 U-Net 对不同肿瘤类型的泛化能力不均衡，胶质瘤可能是后续改进的重点。

#### 10.2 按成像视角分组

测试集结果：

| 视角 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| axial | 346 | 0.7907 | 0.7059 |
| coronal | 257 | 0.7955 | 0.7060 |
| sagittal | 257 | 0.7912 | 0.7093 |

三个视角的结果比较接近，说明 A1 对不同成像平面的适应性较稳定。

#### 10.3 按肿瘤大小分组

测试集结果：

| 肿瘤大小 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| small < 1% | 346 | 0.7589 | 0.6649 |
| medium 1%-5% | 454 | 0.8142 | 0.7334 |
| large > 5% | 60 | 0.8190 | 0.7492 |

小肿瘤组的 Dice 和 IoU 低于中、大肿瘤，说明普通 U-Net 对小目标仍然更容易出现漏分或错分。这也支持后续路线 A 中继续做 Attention U-Net、Boundary Loss 和小目标分析。

输出文件：

```text
outputs/a1/full/eval_test/metrics.csv
outputs/a1/full/eval_test/per_sample_metrics.csv
outputs/a1/full/eval_test/group_metrics.csv
outputs/a1/full/eval_test/test_examples_worst_best.png
outputs/a1/full/eval_test/group_visuals/
```

其中 `group_visuals/` 中保存了按组划分的最差/最好样本可视化：

```text
test_tumor_glioma_worst_best.png
test_tumor_meningioma_worst_best.png
test_tumor_pituitary_worst_best.png
test_view_axial_worst_best.png
test_view_coronal_worst_best.png
test_view_sagittal_worst_best.png
test_size_small_lt1pct_worst_best.png
test_size_medium_1to5pct_worst_best.png
test_size_large_gt5pct_worst_best.png
```

---

## A2：Attention U-Net

状态：已完成。

### 1. 实验目的

A2 的目标是在 A1 普通 U-Net baseline 的基础上，引入 Attention Gate，构建 Attention U-Net。

本步骤主要验证：

- 在 skip connection 中加入注意力机制后，模型是否能更关注肿瘤区域；
- Attention U-Net 是否能减少背景区域对分割结果的干扰；
- Attention U-Net 相比普通 U-Net 是否能提升 Dice 和 IoU。

---

### 2. 模型方法

A2 使用 Attention U-Net：

```text
Attention U-Net + BCE Loss + Dice Loss
```

与普通 U-Net 相比，Attention U-Net 的主要变化是：

```text
Encoder feature → Attention Gate → concat → Decoder
```

也就是在 encoder 和 decoder 的跳跃连接中加入注意力门控，让模型自动筛选更重要的空间区域。

损失函数：

```text
BCE Loss + Dice Loss
```

评价指标：

```text
Dice
IoU
```

---

### 3. 运行命令

正式 A2 使用 GPU 运行，命令如下：

```bash
/home/wxy/python_project/.venv/bin/python scripts/train_a2_attention_unet.py --image-size 128 --base-channels 16 --batch-size 8 --epochs 20 --out-dir outputs/a2/full
```

运行设备：

```text
device = cuda
GPU = NVIDIA GeForce RTX 3060 Laptop GPU
```

---

### 4. 实验结果

训练设置：

```text
model         = Attention U-Net
loss          = BCE Loss + Dice Loss
image_size    = 128
base_channels = 16
batch_size    = 8
epochs        = 20
train_pairs   = 3147
val_pairs     = 786
device        = cuda
```

最好验证集结果出现在第 19 个 epoch：

```text
best_epoch = 19
val_loss   = 0.1263
val_dice   = 0.7851
val_iou    = 0.7011
```

最后一轮结果：

```text
epoch      = 20
train_dice = 0.7883
train_iou  = 0.7027
val_dice   = 0.7663
val_iou    = 0.6768
```

验证集 Dice 变化趋势：

```text
epoch 1  : val_dice = 0.2865
epoch 5  : val_dice = 0.6113
epoch 10 : val_dice = 0.7430
epoch 15 : val_dice = 0.7142
epoch 19 : val_dice = 0.7851
epoch 20 : val_dice = 0.7663
```

---

### 5. 与 A1 对比

| 实验 | 模型 | val Dice | val IoU | test Dice | test IoU |
|---|---|---:|---:|---:|---:|
| A1 | U-Net | 0.7938 | 0.7082 | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 0.7851 | 0.7011 | 0.7937 | 0.7087 |

从独立测试集结果看，A2 的 Dice 略高于 A1：

```text
test Dice 提升 = 0.7937 - 0.7930 = 0.0007
```

但提升幅度非常小，说明单独加入 Attention Gate 对整体平均指标帮助有限。

---

### 6. 输出文件

A2 正式实验结果保存在：

```text
outputs/a2/full/
```

主要文件：

```text
outputs/a2/full/history.csv
outputs/a2/full/checkpoints/best_attention_unet.pt
outputs/a2/full/figures/sample_prediction.png
outputs/a2/full/figures/training_curves.png
outputs/a2/full/readable_checkpoint/
outputs/a2/full/eval_val/
outputs/a2/full/eval_test/
outputs/a2/overfit_8/
```

---

### 7. 阶段结论

A2 已经成功完成。Attention U-Net 在验证集上取得：

```text
best val_dice = 0.7851
best val_iou  = 0.7011
test_dice     = 0.7937
test_iou      = 0.7087
```

与 A1 普通 U-Net 相比，Attention U-Net 在测试集上略有提升，但幅度很小，说明注意力机制在当前设置下对整体区域重叠指标的帮助有限。

---

### 8. 小数据集过拟合测试

为了验证 A2 的 Attention U-Net 结构、损失函数和训练流程是否能正常学习，额外进行了 8 张图的小数据集过拟合 sanity check。

实验设置：

```text
model       = Attention U-Net
num_samples = 8
image_size  = 128
batch_size  = 8
epochs      = 200
augment     = False
device      = cuda
```

运行命令：

```bash
/home/wxy/python_project/.venv/bin/python scripts/overfit_a2_attention_unet.py --image-size 128 --base-channels 16 --batch-size 8 --num-samples 8 --epochs 200 --out-dir outputs/a2/overfit_8
```

最终结果：

```text
epoch      = 200
train_loss = 0.4433
train_dice = 0.9795
train_iou  = 0.9604
eval_loss  = 0.4414
eval_dice  = 0.9912
eval_iou   = 0.9828
```

输出文件：

```text
outputs/a2/overfit_8/history.csv
outputs/a2/overfit_8/best_overfit_attention_unet.pt
outputs/a2/overfit_8/overfit_examples.png
outputs/a2/overfit_8/overfit_curves.png
```

结论：

```text
Attention U-Net 可以在 8 张训练图像上过拟合到 Dice ≈ 0.99。
```

这说明 A2 模型结构和训练流程能够正常学习给定 image-mask 的对应关系。

---

### 9. 独立测试集评估与分组分析

使用 A2 的 `best_attention_unet.pt` 在 `segmentation_task/test` 上进行独立测试集评估。

运行命令：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint outputs/a2/full/checkpoints/best_attention_unet.pt --model attention_unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a2/full/eval_test
```

测试集整体结果：

```text
n_samples = 860
test_dice = 0.7937
test_iou  = 0.7087
```

#### 9.1 按肿瘤类型分组

测试集结果：

| 肿瘤类型 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| glioma | 254 | 0.6360 | 0.5317 |
| meningioma | 306 | 0.8926 | 0.8313 |
| pituitary | 300 | 0.8246 | 0.7310 |

A2 对垂体瘤较 A1 有提升，但对胶质瘤和脑膜瘤略低于 A1。

#### 9.2 按成像视角分组

测试集结果：

| 视角 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| axial | 346 | 0.7878 | 0.7023 |
| coronal | 257 | 0.7941 | 0.7070 |
| sagittal | 257 | 0.7991 | 0.7161 |

A2 在矢状位上的表现略高。

#### 9.3 按肿瘤大小分组

测试集结果：

| 肿瘤大小 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| small < 1% | 346 | 0.7434 | 0.6501 |
| medium 1%-5% | 454 | 0.8232 | 0.7423 |
| large > 5% | 60 | 0.8516 | 0.7799 |

A2 对中、大肿瘤的效果较好，但小肿瘤组仍然明显较低，说明小目标仍是后续 Boundary Loss 和小目标分析的重点。

输出文件：

```text
outputs/a2/full/eval_test/metrics.csv
outputs/a2/full/eval_test/per_sample_metrics.csv
outputs/a2/full/eval_test/group_metrics.csv
outputs/a2/full/eval_test/test_examples_worst_best.png
outputs/a2/full/eval_test/group_visuals/
```

其中 `group_visuals/` 中保存了按组划分的最差/最好样本可视化：

```text
test_tumor_glioma_worst_best.png
test_tumor_meningioma_worst_best.png
test_tumor_pituitary_worst_best.png
test_view_axial_worst_best.png
test_view_coronal_worst_best.png
test_view_sagittal_worst_best.png
test_size_small_lt1pct_worst_best.png
test_size_medium_1to5pct_worst_best.png
test_size_large_gt5pct_worst_best.png
```

后续需要继续完成 A3 和 A4，观察 Boundary Loss 是否能进一步改善肿瘤边界分割效果。

---

## A3：U-Net + Boundary Loss

状态：未开始。

计划：

```text
U-Net + BCE Loss + Dice Loss + Boundary Loss
```

目标：

```text
观察边界损失是否能改善肿瘤轮廓分割效果。
```

---

## A4：Attention U-Net + Boundary Loss

状态：未开始。

计划：

```text
Attention U-Net + BCE Loss + Dice Loss + Boundary Loss
```

目标：

```text
作为路线 A 的核心模型，与 A1/A2/A3 进行对比。
```

---

## A5：小目标分组分析

状态：未开始。

计划：

根据真实 mask 面积占图像比例，将样本分为：

```text
小肿瘤：< 1%
中肿瘤：1% - 5%
大肿瘤：> 5%
```

分别计算 Dice 和 IoU。

目标：

```text
分析模型在不同大小肿瘤上的分割表现，重点观察小肿瘤是否容易漏分。
```
