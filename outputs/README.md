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
