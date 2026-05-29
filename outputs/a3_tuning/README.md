# A3 Tuning

本目录保存 A3 的输入分辨率、small tumor oversampling、threshold sweep、Focal Tversky 和 Boundary Loss 权重调参结果，不覆盖原始 `outputs/a3/full/`。

A3 tuning 是基于 A5 选择出的原始 A3 进行的补充优化探索。该阶段复用已有 checkpoint，在 validation set 上比较不同 boundary weight、oversampling、loss 变体、image size 和 threshold，并根据 validation per-sample mean Dice 选择候选配置。固定该候选配置后，仅在 test set 上进行一次 final test evaluation；test set 不用于 tuning 过程中的模型或 threshold 选择。

最终报告主结果仍采用原始 A3（U-Net + Boundary Loss）：test Dice = 0.8075，test IoU = 0.7271。A3 tuning 的固定候选配置 final test Dice = 0.7986，final test IoU = 0.7164，未超过原始 A3，因此本目录结果作为 supplemental tuning experiment 和 validation-based configuration search，不替代原始 A3。

`legacy_test_exploration_not_for_selection.csv` 是历史测试记录，仅保留作结果追溯，不用于模型选择或调参。
`threshold_sweep/` 是历史阈值扫描记录，也不用于 threshold 选择；当前 threshold 选择只看 `threshold_sweep_val.csv`。

## Decision Rules

| A3 tuning 内容 | 判断依据 | 结果文件 |
|---|---|---|
| boundary_weight=0.05/0.1/0.2/0.3/0.5 | val Dice / val IoU | `summary_val.csv` |
| image_size=128/192/256 | val Dice / val IoU | `summary_val.csv` |
| small oversampling 权重 | val Dice / val IoU / small tumor val 指标 | `summary_val.csv`, `summary_val_group_metrics.csv` |
| Focal Tversky 等 loss variant | val Dice / val IoU | `summary_val.csv` |
| threshold=0.30/0.35/0.40/0.45/0.50/0.55/0.60 | val Dice / val IoU | `threshold_sweep_val.csv` |
| fixed tuning candidate 泛化表现 | test Dice / test IoU | `final_test.csv` |

按 validation per-sample mean Dice 的当前综合选择：

| tuning family | validation 选择 | threshold | val Dice | val IoU | small val Dice | small val IoU |
|---|---|---:|---:|---:|---:|---:|
| boundary weight | A3_boundary_w03 | 0.30 | 0.8052 | 0.7241 | 0.7685 | 0.6827 |
| image size | A3_original_boundary_w02 | 0.50 | 0.8033 | 0.7216 | 0.7549 | 0.6668 |
| oversampling | A3_original_boundary_w02 | 0.50 | 0.8033 | 0.7216 | 0.7549 | 0.6668 |
| loss variant | A3_original_boundary_w02 | 0.50 | 0.8033 | 0.7216 | 0.7549 | 0.6668 |

small oversampling w=3 的 small val Dice / IoU 为 0.7716 / 0.6879，高于 A3 original，但 overall val Dice / IoU 为 0.7932 / 0.7103，因此不作为 validation-based tuning 综合候选配置。

## Validation-Based Candidate Selection

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

候选配置选择规则：

```text
selection split  = validation set
selection metric = per-sample mean val Dice
tie breakers     = val IoU, val Precision, val Recall
final test       = fixed tuning candidate + fixed threshold，仅用于泛化评估
```

Validation set 选出的 tuning 候选配置：

| selected candidate | threshold | val Dice | val IoU | val Precision | val Recall |
|---|---:|---:|---:|---:|---:|
| A3_boundary_w03 | 0.30 | 0.8052 | 0.7241 | 0.8335 | 0.8205 |

固定该 tuning 候选配置后的 final test 结果：

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

`threshold_sweep_val.csv` 保存 validation set 上的 threshold sweep 结果。tuning threshold 由 validation per-sample mean Dice 选择，当前为 `0.30`。

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
