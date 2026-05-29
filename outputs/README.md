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

A5 不重新训练模型，汇总 A1-A4 已有测试集整体指标和分组指标。A1-A4 训练阶段只使用 train 数据，并从 train 内部划分 validation set 保存 best checkpoint；test set 不参与训练，也不参与 checkpoint 保存。A5 使用固定 checkpoint 的 `eval_test` 结果做综合比较，并选择原始 A3 作为后续调参基础模型。

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

## A3 Tuning：validation-based selection

```text
outputs/a3_tuning/
```

该目录保存 A3 tuning 的已有 checkpoint、validation 选择结果和固定候选配置后的 test 评估结果，未覆盖原始 A3。

A3 tuning 是基于 A5 选择出的原始 A3 进行的补充优化探索。该阶段复用已有 checkpoint，在 validation set 上比较不同 boundary weight、oversampling、loss 变体、image size 和 threshold，并根据 validation per-sample mean Dice 选择候选配置。固定该候选配置后，仅在 test set 上进行一次 final test evaluation；test set 不用于 tuning 过程中的模型或 threshold 选择。

判断规则：

```text
boundary_weight / image_size / loss variant：val Dice / val IoU
small oversampling：val Dice / val IoU / small tumor val 指标
threshold：validation threshold sweep
fixed tuning candidate：test Dice / test IoU
```

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
test set      : 固定 tuning candidate 后进行一次 final test evaluation
metric        : per-sample mean
```

Validation set 选出的 tuning 候选配置：

| candidate | threshold | val Dice | val IoU | val Precision | val Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.8052 | 0.7241 | 0.8335 | 0.8205 |

Tuning candidate final test 结果：

| candidate | threshold | test Dice | test IoU | test Precision | test Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.7986 | 0.7164 | 0.8235 | 0.8311 |

最终主结果仍采用原始 A3：

```text
model = A3 original: U-Net + Boundary Loss
test_dice = 0.8075
test_iou  = 0.7271
```

A3 tuning 的固定候选配置 final test 未超过原始 A3，因此 A3 tuning 作为补充分析和后续优化探索，不替代原始 A3 作为最终主结果。

## Final Model

```text
outputs/final_model/
```

该目录根据已有结果文件自动汇总 final model 选择，并保存 final model 在 `segmentation_task/test` 上的最终推理结果、分组统计和可视化展示。不复制 `.pt` 权重文件，只引用原 checkpoint 路径。

主要文件：

```text
README.md                 final model 选择说明、推理结果和可视化说明
final_model.json          机器可读的 final model 选择依据、候选指标和最终选择
final_test_summary.csv    final model 在 test split 上的整体指标
per_sample_metrics.csv    每张测试图的 Dice / IoU / Precision / Recall
group_metrics.csv         tumor / view / size 分组指标
selected_examples.csv     各可视化图选中的 best / typical / worst 样本
figures/                  overall / tumor / view / size 可视化图片
```

选择依据来自 `outputs/final_model/final_model.json`：A1-A4 主实验先按 independent test Dice 选择，接近或并列时再看 test IoU；随后将 A3 tuning final candidate 与主实验最佳模型比较。当前 final model 仍为原始 A3（U-Net + Boundary Loss），checkpoint = `outputs/a3/full/checkpoints/best_unet_boundary.pt`，threshold = 0.50，test Dice = 0.8075，test IoU = 0.7271。A3 tuning final candidate 未超过原始 A3，因此作为 supplemental tuning experiment 保留。

Final inference 已完成：

```text
split = segmentation_task/test
per-sample mean Dice = 0.8069
per-sample mean IoU  = 0.7263
```

生成的最终展示图片：

```text
figures/overall_examples.png
figures/tumor_glioma_examples.png
figures/tumor_meningioma_examples.png
figures/tumor_pituitary_examples.png
figures/view_axial_examples.png
figures/view_coronal_examples.png
figures/view_sagittal_examples.png
figures/size_small_examples.png
figures/size_medium_examples.png
figures/size_large_examples.png
```
