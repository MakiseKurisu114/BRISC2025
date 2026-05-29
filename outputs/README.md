# Outputs Index

本目录保存路线 A 的训练、评估、可视化和模型文件。

## A1：U-Net Baseline

```text
outputs/a1/full/
```

正式 A1 baseline 结果。

主要文件：

```text
history.csv                              每个 epoch 的 loss / Dice / IoU
figures/training_curves.png              训练曲线
checkpoints/best_unet.pt                 A1 最佳模型权重
readable_checkpoint/                     可读模型结构和参数摘要
eval_val/                                固定验证集评估
eval_test/                               独立测试集评估
```

`eval_test/group_visuals/` 保存 A1 在测试集上的分组可视化：

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

```text
outputs/a1/overfit_8/
```

A1 小数据集过拟合 sanity check。

主要文件：

```text
history.csv
overfit_curves.png
overfit_examples.png
best_overfit_unet.pt
```

## A2：Attention U-Net

```text
outputs/a2/full/
```

正式 A2 Attention U-Net 结果。

主要文件：

```text
history.csv
figures/training_curves.png
figures/sample_prediction.png
checkpoints/best_attention_unet.pt
readable_checkpoint/
eval_val/
eval_test/
```

`eval_test/group_visuals/` 保存 A2 在测试集上的分组可视化：

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

```text
outputs/a2/overfit_8/
```

A2 小数据集过拟合 sanity check。

主要文件：

```text
history.csv
overfit_curves.png
overfit_examples.png
best_overfit_attention_unet.pt
```

## A3：U-Net + Boundary Loss

```text
outputs/a3/full/
```

正式 A3 U-Net + Boundary Loss 结果。

主要文件：

```text
history.csv
figures/training_curves.png
figures/sample_prediction.png
checkpoints/best_unet_boundary.pt
readable_checkpoint/
eval_val/
eval_test/
```

`eval_test/` 保存 A3 在独立测试集上的整体评估、逐样本指标、分组指标和最差/最好样本可视化：

```text
metrics.csv
per_sample_metrics.csv
group_metrics.csv
test_examples_worst_best.png
```

`eval_test/group_visuals/` 保存 A3 在测试集上的分组可视化：

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

```text
outputs/a3/overfit_8/
```

A3 小数据集过拟合 sanity check。

主要文件：

```text
history.csv
overfit_curves.png
overfit_examples.png
best_overfit_unet_boundary.pt
```

A3 独立测试集整体结果：

```text
n_samples = 860
test_dice = 0.8075
test_iou  = 0.7271
```

## A4：Attention U-Net + Boundary Loss

```text
outputs/a4/full/
```

正式 A4 Attention U-Net + Boundary Loss 结果。

主要文件：

```text
history.csv
figures/training_curves.png
figures/sample_prediction.png
checkpoints/best_attention_unet_boundary.pt
readable_checkpoint/
eval_val/
eval_test/
```

`eval_test/` 保存 A4 在独立测试集上的整体评估、逐样本指标、分组指标和最差/最好样本可视化：

```text
metrics.csv
per_sample_metrics.csv
group_metrics.csv
test_examples_worst_best.png
```

`eval_test/group_visuals/` 保存 A4 在测试集上的分组可视化：

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

```text
outputs/a4/overfit_8/
```

A4 小数据集过拟合 sanity check。

主要文件：

```text
history.csv
overfit_curves.png
overfit_examples.png
best_overfit_attention_unet_boundary.pt
```

A4 独立测试集整体结果：

```text
n_samples = 860
test_dice = 0.7923
test_iou  = 0.7093
```

## A5：A1-A4 分组综合分析

```text
outputs/a5/summary/
```

A5 不重新训练模型，汇总 A1-A4 已有测试集整体指标和分组指标。

主要文件：

```text
overall_test_metrics.csv                 A1-A4 独立测试集整体 Dice / IoU
group_metrics_long.csv                   A1-A4 所有分组长表
best_by_group.csv                        每个分组 Dice / IoU 最优模型
tumor_mean_dice_pivot.csv                肿瘤类型 Dice 对比透视表
tumor_mean_iou_pivot.csv                 肿瘤类型 IoU 对比透视表
view_mean_dice_pivot.csv                 成像视角 Dice 对比透视表
view_mean_iou_pivot.csv                  成像视角 IoU 对比透视表
size_group_mean_dice_pivot.csv           肿瘤大小 Dice 对比透视表
size_group_mean_iou_pivot.csv            肿瘤大小 IoU 对比透视表
README.md                                A5 自动生成摘要
```

`figures/` 保存 A5 对比图：

```text
overall_test_metrics.png
tumor_dice_comparison.png
tumor_iou_comparison.png
view_dice_comparison.png
view_iou_comparison.png
size_group_dice_comparison.png
size_group_iou_comparison.png
```

A5 核心结论：

```text
整体测试集最优：A3 U-Net + Boundary Loss
test_dice = 0.8075
test_iou  = 0.7271
```

分组层面：

```text
glioma / meningioma / axial / coronal / sagittal / medium tumor：A3 最优
small tumor：A1 最优
large tumor Dice：A2 最优
large tumor IoU：A3 最优
```

## A3 Tuning：输入分辨率调参

```text
outputs/a3_tuning/
```

该目录保存 A3 在更高输入分辨率下的调参结果，未覆盖原始 A3。

实验设置：

```text
model           = U-Net
loss            = BCE Loss + Dice Loss + Boundary Loss
boundary_weight = 0.2
base_channels   = 16
epochs          = 20
```

结果：

| 实验 | image_size | batch_size | test Dice | test IoU | small Dice | small IoU |
|---|---:|---:|---:|---:|---:|---:|
| A3 original | 128 | 8 | 0.8075 | 0.7271 | 0.7542 | 0.6638 |
| A3 image_size 192 | 192 | 8 | 0.7809 | 0.6989 | 0.7447 | 0.6555 |
| A3 image_size 256 | 256 | 8 | 0.7744 | 0.6893 | 0.7414 | 0.6521 |
| A3 small oversampling w=1.5 | 128 | 8 | 0.7826 | 0.7020 | 0.7575 | 0.6699 |
| A3 small oversampling w=2.0 | 128 | 8 | 0.7974 | 0.7153 | 0.7487 | 0.6558 |
| A3 small oversampling w=3 | 128 | 8 | 0.7886 | 0.7024 | 0.7811 | 0.6890 |
| A3 Focal Tversky w=0.2 | 128 | 8 | 0.7975 | 0.7140 | 0.7489 | 0.6544 |
| A3 Boundary w=0.05 | 128 | 8 | 0.7900 | 0.7062 | 0.7734 | 0.6799 |
| A3 Boundary w=0.1 | 128 | 8 | 0.8017 | 0.7180 | 0.7525 | 0.6596 |
| A3 Boundary w=0.3 | 128 | 8 | 0.7991 | 0.7172 | 0.7506 | 0.6574 |
| A3 Boundary w=0.5 | 128 | 8 | 0.8120 | 0.7319 | 0.7644 | 0.6704 |

结论：

```text
单纯提高 image_size 到 192/256 没有改善 small tumor，整体性能反而下降。
small tumor oversampling 存在 tradeoff：w=3.0 的 small 组最好，w=2.0 的 overall 折中最好。
threshold 0.30-0.50 扫描显示，降低阈值没有改善 small tumor。对新最佳 Boundary w=0.5，threshold=0.60 可带来轻微提升。
Focal Tversky w=0.2 没有提升 small tumor，整体 test Dice / IoU 也低于 A3 original。
Boundary w=0.5 是当前 overall 最优设置，test Dice / IoU 超过 A3 original，并小幅提升 small tumor。
```
