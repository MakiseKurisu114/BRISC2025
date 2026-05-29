# A5 分组综合分析摘要

A5-val 汇总 A1-A4 已有验证集结果，不重新训练模型。

A5-val 用于模型比较、分组分析和后续调参方向判断；test set 只用于最终固定配置后的泛化评估。

## 验证集整体指标

| experiment | model | n_samples | val_dice | val_iou |
| --- | --- | --- | --- | --- |
| A1 | U-Net | 786 | 0.7938 | 0.7082 |
| A2 | Attention U-Net | 786 | 0.7851 | 0.7011 |
| A3 | U-Net + Boundary Loss | 786 | 0.8043 | 0.7228 |
| A4 | Attention U-Net + Boundary Loss | 786 | 0.7934 | 0.7110 |

## 各分组最优模型

| group_type | group_label | n | best_dice_experiment | best_dice | best_iou_experiment | best_iou |
| --- | --- | --- | --- | --- | --- | --- |
| tumor | glioma | 231 | A3 | 0.6439 | A3 | 0.5398 |
| tumor | meningioma | 259 | A3 | 0.9114 | A3 | 0.8588 |
| tumor | pituitary | 296 | A3 | 0.8331 | A3 | 0.7436 |
| view | axial | 251 | A3 | 0.7864 | A3 | 0.7064 |
| view | coronal | 267 | A1 | 0.8120 | A1 | 0.7229 |
| view | sagittal | 264 | A3 | 0.8246 | A3 | 0.7503 |
| size_group | large > 5% | 20 | A1 | 0.7919 | A1 | 0.7039 |
| size_group | medium 1%-5% | 458 | A3 | 0.8377 | A3 | 0.7604 |
| size_group | small < 1% | 294 | A1 | 0.7662 | A1 | 0.6716 |

## 主要结论

- 验证集整体最优模型是 A3（U-Net + Boundary Loss），Dice = 0.8043，IoU = 0.7228。
- glioma 仍然是最难分割的肿瘤类型。
- small tumor 仍然比 medium / large tumor 更困难。
- 当前控制变量对比中，Boundary Loss 是最明确有效的改进。
- 当前超参数下，Attention U-Net + Boundary Loss 没有超过 U-Net + Boundary Loss。
- 后续 A3 tuning 以 validation-based selection 选择配置和 threshold。
