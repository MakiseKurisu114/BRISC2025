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

后续已继续完成 A3 和 A4，可在最终分组分析中统一比较 Boundary Loss 和 Attention Gate 的影响。

---

## A3：U-Net + Boundary Loss

状态：已完成 A3 训练、验证集评估、独立测试集评估、小数据集过拟合和分组可视化。

### 1. 实验目的

A3 在 A1 普通 U-Net 的基础上加入 Boundary Loss，不改变模型结构，重点观察边界约束是否能提升分割效果。

与 A1/A2 的区别：

```text
U-Net + BCE Loss + Dice Loss + Boundary Loss
```

其中 Boundary Loss 用预测 mask 和真实 mask 的边界图计算 Dice 类型损失，使模型更关注肿瘤轮廓区域。

---

### 2. 运行命令

```bash
/home/wxy/python_project/.venv/bin/python scripts/train_a3_unet_boundary.py --image-size 128 --base-channels 16 --batch-size 8 --epochs 20 --boundary-weight 0.2 --out-dir outputs/a3/full
```

---

### 3. 实验结果

训练设置：

```text
model           = U-Net
loss            = BCE Loss + Dice Loss + Boundary Loss
boundary_weight = 0.2
image_size      = 128
base_channels   = 16
batch_size      = 8
epochs          = 20
train_pairs     = 3147
val_pairs       = 786
device          = cuda
```

最好验证集结果出现在第 18 个 epoch：

```text
best_epoch = 18
val_loss   = 0.1795
val_dice   = 0.8043
val_iou    = 0.7228
```

最后一轮结果：

```text
epoch      = 20
train_dice = 0.7977
train_iou  = 0.7148
val_dice   = 0.7663
val_iou    = 0.6900
```

验证集 Dice 变化趋势：

```text
epoch 1  : val_dice = 0.5711
epoch 5  : val_dice = 0.6731
epoch 10 : val_dice = 0.7730
epoch 15 : val_dice = 0.7896
epoch 18 : val_dice = 0.8043
epoch 20 : val_dice = 0.7663
```

---

### 4. 与 A1/A2 对比

| 实验 | 模型 | val Dice | val IoU | test Dice | test IoU |
|---|---|---:|---:|---:|---:|
| A1 | U-Net | 0.7938 | 0.7082 | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 0.7851 | 0.7011 | 0.7937 | 0.7087 |
| A3 | U-Net + Boundary Loss | 0.8043 | 0.7228 | 0.8075 | 0.7271 |

A3 在验证集和独立测试集上均超过 A1/A2，说明边界损失在当前实验设置下带来了正向提升。

---

### 5. 小数据集过拟合测试

运行命令：

```bash
/home/wxy/python_project/.venv/bin/python scripts/overfit_a3_unet_boundary.py --image-size 128 --base-channels 16 --batch-size 8 --num-samples 8 --epochs 200 --boundary-weight 0.2 --out-dir outputs/a3/overfit_8
```

最终结果：

```text
epoch      = 200
train_loss = 0.4183
train_dice = 0.9881
train_iou  = 0.9767
eval_loss  = 0.4186
eval_dice  = 0.9888
eval_iou   = 0.9779
```

结论：

```text
U-Net + Boundary Loss 可以在 8 张训练图像上过拟合到 Dice ≈ 0.99。
```

---

### 6. 验证集分组分析

验证集整体结果：

```text
n_samples = 786
val_dice  = 0.8043
val_iou   = 0.7228
```

按肿瘤类型：

| 肿瘤类型 | 样本数 | Dice | IoU |
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

| 肿瘤大小 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| small < 1% | 306 | 0.7549 | 0.6668 |
| medium 1%-5% | 458 | 0.8377 | 0.7604 |
| large > 5% | 22 | 0.7599 | 0.6778 |

---

### 7. 输出文件

```text
outputs/a3/full/history.csv
outputs/a3/full/checkpoints/best_unet_boundary.pt
outputs/a3/full/figures/sample_prediction.png
outputs/a3/full/figures/training_curves.png
outputs/a3/full/readable_checkpoint/
outputs/a3/full/eval_val/
outputs/a3/full/eval_test/
outputs/a3/overfit_8/
```

---

### 8. 独立测试集评估与分组分析

使用 A3 的 `best_unet_boundary.pt` 在 `segmentation_task/test` 上进行独立测试集评估。

运行命令：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint outputs/a3/full/checkpoints/best_unet_boundary.pt --model unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a3/full/eval_test

/home/wxy/python_project/.venv/bin/python scripts/generate_group_visuals.py --checkpoint outputs/a3/full/checkpoints/best_unet_boundary.pt --model unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a3/full/eval_test/group_visuals
```

测试集整体结果：

```text
n_samples = 860
test_dice = 0.8075
test_iou  = 0.7271
device    = cuda
```

#### 8.1 按肿瘤类型分组

测试集结果：

| 肿瘤类型 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| glioma | 254 | 0.6619 | 0.5619 |
| meningioma | 306 | 0.9099 | 0.8552 |
| pituitary | 300 | 0.8245 | 0.7340 |

#### 8.2 按成像视角分组

测试集结果：

| 视角 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| axial | 346 | 0.8016 | 0.7252 |
| coronal | 257 | 0.8041 | 0.7193 |
| sagittal | 257 | 0.8168 | 0.7348 |

#### 8.3 按肿瘤大小分组

测试集结果：

| 肿瘤大小 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| small < 1% | 346 | 0.7542 | 0.6638 |
| medium 1%-5% | 454 | 0.8423 | 0.7667 |
| large > 5% | 60 | 0.8423 | 0.7809 |

A3 在测试集上取得当前 A1/A2/A3 中最高的整体 Dice 和 IoU。分组结果中，小肿瘤仍明显低于中、大肿瘤；胶质瘤仍是三类肿瘤中最难分割的一类。

输出文件：

```text
outputs/a3/full/eval_test/metrics.csv
outputs/a3/full/eval_test/per_sample_metrics.csv
outputs/a3/full/eval_test/group_metrics.csv
outputs/a3/full/eval_test/test_examples_worst_best.png
outputs/a3/full/eval_test/group_visuals/
```

---

## A4：Attention U-Net + Boundary Loss

状态：已完成。

### 1. 实验目的

A4 将 A2 的 Attention U-Net 和 A3 的 Boundary Loss 组合起来，观察注意力机制与边界约束是否能够叠加提升分割效果。

与前面实验的关系：

```text
A1：U-Net
A2：Attention U-Net
A3：U-Net + Boundary Loss
A4：Attention U-Net + Boundary Loss
```

A4 的模型和损失函数为：

```text
Attention U-Net + BCE Loss + Dice Loss + Boundary Loss
```

---

### 2. 运行命令

正式训练：

```bash
/home/wxy/python_project/.venv/bin/python scripts/train_a4_attention_unet_boundary.py --image-size 128 --base-channels 16 --batch-size 8 --epochs 20 --boundary-weight 0.2 --out-dir outputs/a4/full
```

验证集评估：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint outputs/a4/full/checkpoints/best_attention_unet_boundary.pt --model attention_unet --eval-split val --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a4/full/eval_val
```

独立测试集评估：

```bash
/home/wxy/python_project/.venv/bin/python scripts/evaluate_checkpoint.py --checkpoint outputs/a4/full/checkpoints/best_attention_unet_boundary.pt --model attention_unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a4/full/eval_test
```

测试集分组可视化：

```bash
/home/wxy/python_project/.venv/bin/python scripts/generate_group_visuals.py --checkpoint outputs/a4/full/checkpoints/best_attention_unet_boundary.pt --model attention_unet --eval-split test --image-size 128 --base-channels 16 --batch-size 8 --out-dir outputs/a4/full/eval_test/group_visuals
```

小数据集过拟合：

```bash
/home/wxy/python_project/.venv/bin/python scripts/overfit_a4_attention_unet_boundary.py --image-size 128 --base-channels 16 --batch-size 8 --num-samples 8 --epochs 200 --boundary-weight 0.2 --out-dir outputs/a4/overfit_8
```

---

### 3. 实验结果

训练设置：

```text
model           = Attention U-Net
loss            = BCE Loss + Dice Loss + Boundary Loss
boundary_weight = 0.2
image_size      = 128
base_channels   = 16
batch_size      = 8
epochs          = 20
train_pairs     = 3147
val_pairs       = 786
device          = cuda
```

最好验证集结果出现在第 18 个 epoch：

```text
best_epoch = 18
val_loss   = 0.1876
val_dice   = 0.7934
val_iou    = 0.7110
```

最后一轮结果：

```text
epoch      = 20
train_loss = 0.1856
train_dice = 0.7965
train_iou  = 0.7133
val_loss   = 0.1894
val_dice   = 0.7918
val_iou    = 0.7072
```

验证集 Dice 变化趋势：

```text
epoch 1  : val_dice = 0.5258
epoch 5  : val_dice = 0.6757
epoch 10 : val_dice = 0.7408
epoch 15 : val_dice = 0.7680
epoch 18 : val_dice = 0.7934
epoch 20 : val_dice = 0.7918
```

---

### 4. 与 A1/A2/A3 对比

| 实验 | 模型 | val Dice | val IoU | test Dice | test IoU |
|---|---|---:|---:|---:|---:|
| A1 | U-Net | 0.7938 | 0.7082 | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 0.7851 | 0.7011 | 0.7937 | 0.7087 |
| A3 | U-Net + Boundary Loss | 0.8043 | 0.7228 | 0.8075 | 0.7271 |
| A4 | Attention U-Net + Boundary Loss | 0.7934 | 0.7110 | 0.7923 | 0.7093 |

A4 可以正常训练，但当前设置下没有超过 A3。说明在本实验配置中，Boundary Loss 对普通 U-Net 的提升最明显，而 Attention Gate 与 Boundary Loss 的简单组合没有带来额外整体收益。

---

### 5. 小数据集过拟合测试

最终结果：

```text
epoch      = 200
train_loss = 0.4724
train_dice = 0.9799
train_iou  = 0.9608
eval_loss  = 0.4720
eval_dice  = 0.9812
eval_iou   = 0.9634
```

结论：

```text
Attention U-Net + Boundary Loss 可以在 8 张训练图像上过拟合到 Dice ≈ 0.98。
```

这说明 A4 的模型结构、损失函数和训练流程能够正常学习给定 image-mask 的对应关系。

---

### 6. 独立测试集评估与分组分析

测试集整体结果：

```text
n_samples = 860
test_dice = 0.7923
test_iou  = 0.7093
device    = cuda
```

#### 6.1 按肿瘤类型分组

测试集结果：

| 肿瘤类型 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| glioma | 254 | 0.6362 | 0.5376 |
| meningioma | 306 | 0.9088 | 0.8505 |
| pituitary | 300 | 0.8036 | 0.7082 |

#### 6.2 按成像视角分组

测试集结果：

| 视角 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| axial | 346 | 0.7959 | 0.7135 |
| coronal | 257 | 0.7987 | 0.7148 |
| sagittal | 257 | 0.7786 | 0.6953 |

#### 6.3 按肿瘤大小分组

测试集结果：

| 肿瘤大小 | 样本数 | Dice | IoU |
|---|---:|---:|---:|
| small < 1% | 346 | 0.7497 | 0.6519 |
| medium 1%-5% | 454 | 0.8218 | 0.7484 |
| large > 5% | 60 | 0.8039 | 0.7321 |

A4 的分组趋势与前面实验一致：胶质瘤仍然低于脑膜瘤和垂体瘤，小肿瘤仍然低于中等肿瘤。相比 A3，A4 在 glioma、pituitary、sagittal 和 large 组上均偏低。

---

### 7. 输出文件

```text
outputs/a4/full/history.csv
outputs/a4/full/checkpoints/best_attention_unet_boundary.pt
outputs/a4/full/figures/sample_prediction.png
outputs/a4/full/figures/training_curves.png
outputs/a4/full/readable_checkpoint/
outputs/a4/full/eval_val/
outputs/a4/full/eval_test/
outputs/a4/full/eval_test/group_visuals/
outputs/a4/overfit_8/
```

---

### 8. 阶段结论

A4 已完成完整训练、验证集评估、独立测试集评估、小数据集过拟合和测试集分组可视化。

主要结论：

- A4 能够正常学习，小数据集过拟合 Dice 达到 0.9812；
- A4 测试集 Dice / IoU 为 0.7923 / 0.7093；
- A4 没有超过 A3，当前路线 A 中整体最优仍是 A3；
- 后续 A5 应重点围绕 A1/A2/A3/A4 的分组差异展开，尤其比较 glioma 和 small tumor。

---

## A5：A1-A4 分组综合分析

状态：已完成。

### 1. 实验目的

A5 不训练新模型，而是汇总 A1/A2/A3/A4 在独立测试集上的整体指标和分组指标，回答以下问题：

- 哪个模型整体最好；
- 哪个模型对不同肿瘤类型最好；
- 哪个模型对不同成像视角最好；
- 哪个模型对小 / 中 / 大肿瘤最好；
- Boundary Loss 和 Attention Gate 在分组层面是否真的带来收益。

---

### 2. 运行命令

```bash
/home/wxy/python_project/.venv/bin/python scripts/summarize_a5_group_analysis.py --out-dir outputs/a5/summary
```

该脚本读取 A1-A4 的测试集 `metrics.csv` 和 `group_metrics.csv`，并生成：

```text
outputs/a5/summary/overall_test_metrics.csv
outputs/a5/summary/group_metrics_long.csv
outputs/a5/summary/best_by_group.csv
outputs/a5/summary/*_pivot.csv
outputs/a5/summary/figures/
outputs/a5/summary/README.md
```

---

### 3. A1-A4 独立测试集整体对比

| 实验 | 模型 | 样本数 | test Dice | test IoU |
|---|---|---:|---:|---:|
| A1 | U-Net | 860 | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 860 | 0.7937 | 0.7087 |
| A3 | U-Net + Boundary Loss | 860 | 0.8075 | 0.7271 |
| A4 | Attention U-Net + Boundary Loss | 860 | 0.7923 | 0.7093 |

整体结论：

```text
A3 是 A1-A4 中整体测试集表现最好的模型。
```

A4 没有超过 A3，说明在当前固定超参数下，Attention Gate 与 Boundary Loss 的组合没有形成叠加收益。

---

### 4. 按肿瘤类型分组

| 肿瘤类型 | 样本数 | A1 Dice | A2 Dice | A3 Dice | A4 Dice | Dice 最优 |
|---|---:|---:|---:|---:|---:|---|
| glioma | 254 | 0.6482 | 0.6360 | 0.6619 | 0.6362 | A3 |
| meningioma | 306 | 0.8998 | 0.8926 | 0.9099 | 0.9088 | A3 |
| pituitary | 300 | 0.8047 | 0.8246 | 0.8245 | 0.8036 | A2 |

| 肿瘤类型 | 样本数 | A1 IoU | A2 IoU | A3 IoU | A4 IoU | IoU 最优 |
|---|---:|---:|---:|---:|---:|---|
| glioma | 254 | 0.5446 | 0.5317 | 0.5619 | 0.5376 | A3 |
| meningioma | 306 | 0.8374 | 0.8313 | 0.8552 | 0.8505 | A3 |
| pituitary | 300 | 0.7113 | 0.7310 | 0.7340 | 0.7082 | A3 |

结论：

- glioma 是最难分割的肿瘤类型，所有模型在该组 Dice 都明显低于 meningioma 和 pituitary；
- A3 在 glioma 和 meningioma 上同时取得最高 Dice / IoU；
- pituitary 的 Dice 由 A2 略高，但 IoU 仍是 A3 最好，二者差距极小。

---

### 5. 按成像视角分组

| 视角 | 样本数 | A1 Dice | A2 Dice | A3 Dice | A4 Dice | Dice 最优 |
|---|---:|---:|---:|---:|---:|---|
| axial | 346 | 0.7907 | 0.7878 | 0.8016 | 0.7959 | A3 |
| coronal | 257 | 0.7955 | 0.7941 | 0.8041 | 0.7987 | A3 |
| sagittal | 257 | 0.7912 | 0.7991 | 0.8168 | 0.7786 | A3 |

| 视角 | 样本数 | A1 IoU | A2 IoU | A3 IoU | A4 IoU | IoU 最优 |
|---|---:|---:|---:|---:|---:|---|
| axial | 346 | 0.7059 | 0.7023 | 0.7252 | 0.7135 | A3 |
| coronal | 257 | 0.7060 | 0.7070 | 0.7193 | 0.7148 | A3 |
| sagittal | 257 | 0.7093 | 0.7161 | 0.7348 | 0.6953 | A3 |

结论：

```text
A3 在 axial / coronal / sagittal 三个视角上均为最优。
```

这说明 Boundary Loss 的提升不是只来自某一个特定视角，而是在不同成像方向上都比较稳定。

---

### 6. 按肿瘤大小分组

根据真实 mask 面积占图像比例，将样本分为：

```text
小肿瘤：< 1%
中肿瘤：1% - 5%
大肿瘤：> 5%
```

| 肿瘤大小 | 样本数 | A1 Dice | A2 Dice | A3 Dice | A4 Dice | Dice 最优 |
|---|---:|---:|---:|---:|---:|---|
| small < 1% | 346 | 0.7589 | 0.7434 | 0.7542 | 0.7497 | A1 |
| medium 1%-5% | 454 | 0.8142 | 0.8232 | 0.8423 | 0.8218 | A3 |
| large > 5% | 60 | 0.8190 | 0.8516 | 0.8423 | 0.8039 | A2 |

| 肿瘤大小 | 样本数 | A1 IoU | A2 IoU | A3 IoU | A4 IoU | IoU 最优 |
|---|---:|---:|---:|---:|---:|---|
| small < 1% | 346 | 0.6649 | 0.6501 | 0.6638 | 0.6519 | A1 |
| medium 1%-5% | 454 | 0.7334 | 0.7423 | 0.7667 | 0.7484 | A3 |
| large > 5% | 60 | 0.7492 | 0.7799 | 0.7809 | 0.7321 | A3 |

结论：

- small tumor 仍是难点，A3 整体最优但在 small 组没有超过 A1；
- medium tumor 是 A3 提升最明显的大小组；
- large tumor 样本数只有 60，Dice 最优为 A2，IoU 最优为 A3，解释时应注意样本数较少带来的波动。

---

### 7. 输出文件

```text
outputs/a5/summary/overall_test_metrics.csv
outputs/a5/summary/group_metrics_long.csv
outputs/a5/summary/best_by_group.csv
outputs/a5/summary/tumor_mean_dice_pivot.csv
outputs/a5/summary/tumor_mean_iou_pivot.csv
outputs/a5/summary/view_mean_dice_pivot.csv
outputs/a5/summary/view_mean_iou_pivot.csv
outputs/a5/summary/size_group_mean_dice_pivot.csv
outputs/a5/summary/size_group_mean_iou_pivot.csv
outputs/a5/summary/README.md
outputs/a5/summary/figures/overall_test_metrics.png
outputs/a5/summary/figures/tumor_dice_comparison.png
outputs/a5/summary/figures/tumor_iou_comparison.png
outputs/a5/summary/figures/view_dice_comparison.png
outputs/a5/summary/figures/view_iou_comparison.png
outputs/a5/summary/figures/size_group_dice_comparison.png
outputs/a5/summary/figures/size_group_iou_comparison.png
```

---

### 8. 最终阶段结论

```text
A3：U-Net + Boundary Loss 是当前最优模型。
```

更具体地说：

- Attention U-Net 单独使用时，相比 U-Net 只有很小提升；
- Boundary Loss 单独加入 U-Net 时，带来最明确的整体提升；
- Attention U-Net + Boundary Loss 没有超过 U-Net + Boundary Loss；
- glioma 始终是最难分割的肿瘤类型；
- small tumor 并未因为 Boundary Loss 明显改善，后续若调参，应优先关注小目标；
- A5 支持后续做少量调参，例如 A3 的 `boundary_weight` 或针对 small tumor 的增强策略。
