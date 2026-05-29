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

## A5-val：A1-A4 验证集分组综合分析

```text
outputs/a5/summary/
```

A5-val 不重新训练模型，汇总 A1-A4 已有验证集整体指标和分组指标，用于模型比较、分组分析和后续调参方向判断。test set 只用于最终固定配置后的泛化评估。

主要文件：

```text
overall_val_metrics.csv                  A1-A4 验证集整体 Dice / IoU
group_val_metrics_long.csv               A1-A4 验证集所有分组长表
best_by_group_val.csv                    验证集每个分组 Dice / IoU 最优模型
tumor_mean_dice_pivot.csv                肿瘤类型 Dice 对比透视表
tumor_mean_iou_pivot.csv                 肿瘤类型 IoU 对比透视表
view_mean_dice_pivot.csv                 成像视角 Dice 对比透视表
view_mean_iou_pivot.csv                  成像视角 IoU 对比透视表
size_group_mean_dice_pivot.csv           肿瘤大小 Dice 对比透视表
size_group_mean_iou_pivot.csv            肿瘤大小 IoU 对比透视表
README.md                                A5 自动生成摘要
```

`figures/` 保存 A5-val 对比图：

```text
overall_val_metrics.png
tumor_dice_comparison.png
tumor_iou_comparison.png
view_dice_comparison.png
view_iou_comparison.png
size_group_dice_comparison.png
size_group_iou_comparison.png
```

A5-val 核心结论：

```text
验证集整体最优：A3 U-Net + Boundary Loss
val_dice = 0.8043
val_iou  = 0.7228
```

分组层面：

```text
glioma / meningioma / pituitary / axial / sagittal / medium tumor：A3 最优
small tumor：A1 最优
coronal / large tumor：A1 最优
```

## A3 Tuning：validation-based selection

```text
outputs/a3_tuning/
```

该目录保存 A3 tuning 的已有 checkpoint、validation 选择结果和固定最终配置后的 test 评估结果，未覆盖原始 A3。

A3 tuning 阶段复用已有 checkpoint，在 validation set 上比较不同 boundary weight、oversampling、loss 变体、image size 和 threshold，并根据 validation per-sample mean Dice 选择最终配置。固定最终配置后，仅在 test set 上进行一次最终评估。

基础实验设置：

```text
model           = U-Net
loss            = BCE Loss + Dice Loss + Boundary Loss
boundary_weight = 0.2
base_channels   = 16
epochs          = 20
```

主要结果文件：

```text
summary_val.csv
threshold_sweep_val.csv
summary_val_group_metrics.csv
final_selection.json
final_test.csv
final_test_group_metrics.csv
final_test_per_sample_metrics.csv
final_test_examples_worst_best.png
```

流程说明：

```text
validation set: 选择模型设置和 threshold
test set      : 固定最终配置后进行一次最终评估
metric        : per-sample mean
```

Validation set 选出的最终配置：

| candidate | threshold | val Dice | val IoU | val Precision | val Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.8052 | 0.7241 | 0.8335 | 0.8205 |

Final test 结果：

| candidate | threshold | test Dice | test IoU | test Precision | test Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.7986 | 0.7164 | 0.8235 | 0.8311 |
