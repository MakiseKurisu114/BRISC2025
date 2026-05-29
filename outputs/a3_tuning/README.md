# A3 Tuning

本目录保存 A3 的输入分辨率、small tumor oversampling、threshold sweep、Focal Tversky 和 Boundary Loss 权重调参结果，不覆盖原始 `outputs/a3/full/`。

A3 tuning 阶段复用已有 checkpoint，在 validation set 上比较不同 boundary weight、oversampling、loss 变体、image size 和 threshold，并根据 validation per-sample mean Dice 选择最终配置。固定最终配置后，仅在 test set 上进行一次最终评估。

`legacy_test_exploration_not_for_selection.csv` 是历史测试记录，仅保留作结果追溯，不用于模型选择或调参。

## Final Selection

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

选择规则：

```text
selection split  = validation set
selection metric = per-sample mean val Dice
tie breakers     = val IoU, val Precision, val Recall
final test       = fixed candidate + fixed threshold
```

Validation set 选出的最终配置：

| selected candidate | threshold | val Dice | val IoU | val Precision | val Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.8052 | 0.7241 | 0.8335 | 0.8205 |

固定该配置后的 final test 结果：

| candidate | threshold | test Dice | test IoU | test Precision | test Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.7986 | 0.7164 | 0.8235 | 0.8311 |

所有 final 指标均为 per-sample mean。

## Candidate Checkpoints

本目录保留以下候选 checkpoint 和对应训练输出，用于 validation-based selection：

```text
image_size_192/full/
image_size_256_bs8/full/
small_oversampling_w15/full/
small_oversampling_w2/full/
small_oversampling_w3/full/
focal_tversky_w02/full/
boundary_w005/full/
boundary_w01/full/
boundary_w03/full/
boundary_w05/full/
```

A3 original checkpoint 作为候选之一参与 validation 选择：

```text
outputs/a3/full/checkpoints/best_unet_boundary.pt
```

## Threshold Sweep

`threshold_sweep_val.csv` 保存 validation set 上的 threshold sweep 结果。最终 threshold 由 validation per-sample mean Dice 选择，当前为 `0.30`。

## Output Files

```text
legacy_test_exploration_not_for_selection.csv
summary_val.csv
threshold_sweep_val.csv
summary_val_group_metrics.csv
final_selection.json
final_test.csv
final_test_group_metrics.csv
final_test_per_sample_metrics.csv
final_test_examples_worst_best.png
image_size_192/full/
image_size_256_bs8/full/
small_oversampling_w3/full/
small_oversampling_w15/full/
small_oversampling_w2/full/
focal_tversky_w02/full/
boundary_w005/full/
boundary_w01/full/
boundary_w03/full/
boundary_w05/full/
threshold_sweep/
```
