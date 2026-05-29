# Legacy Threshold Sweep

本目录保存早期阈值扫描记录，仅用于结果追溯，不用于 A3 tuning 的 threshold 选择。

当前 A3 tuning 的 threshold 选择必须使用 validation set：

```text
outputs/a3_tuning/threshold_sweep_val.csv
```

当前 validation-based tuning 候选配置：

```text
candidate = A3_boundary_w03
threshold = 0.30
selection split = validation set
selection metric = per-sample mean val Dice
```

这里的 `A3_boundary_w03` 是 validation-based tuning 选出的候选配置，不是最终报告主模型；最终主结果仍采用原始 A3。

固定上述配置后，test set 只用于最终泛化评估：

```text
outputs/a3_tuning/final_test.csv
```

该候选配置的 final test 未超过原始 A3，因此最终报告主结果仍采用原始 A3：test Dice = 0.8075，test IoU = 0.7271。
