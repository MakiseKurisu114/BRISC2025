# A5 分组综合分析摘要

A5 汇总 A1-A4 已有测试集结果，不重新训练模型。

A1-A4 训练阶段只使用 train 数据，并从 train 内部划分 validation set 保存 best checkpoint；A5 使用这些固定 checkpoint 的 test 结果进行综合比较。

## 独立测试集整体指标

| experiment | model | n_samples | test_dice | test_iou |
| --- | --- | --- | --- | --- |
| A1 | U-Net | 860 | 0.7930 | 0.7078 |
| A2 | Attention U-Net | 860 | 0.7937 | 0.7087 |
| A3 | U-Net + Boundary Loss | 860 | 0.8075 | 0.7271 |
| A4 | Attention U-Net + Boundary Loss | 860 | 0.7923 | 0.7093 |

## 各分组最优模型

| group_type | group_label | n | best_dice_experiment | best_dice | best_iou_experiment | best_iou |
| --- | --- | --- | --- | --- | --- | --- |
| tumor | glioma | 254 | A3 | 0.6619 | A3 | 0.5619 |
| tumor | meningioma | 306 | A3 | 0.9099 | A3 | 0.8552 |
| tumor | pituitary | 300 | A2 | 0.8246 | A3 | 0.7340 |
| view | axial | 346 | A3 | 0.8016 | A3 | 0.7252 |
| view | coronal | 257 | A3 | 0.8041 | A3 | 0.7193 |
| view | sagittal | 257 | A3 | 0.8168 | A3 | 0.7348 |
| size_group | large > 5% | 60 | A2 | 0.8516 | A3 | 0.7809 |
| size_group | medium 1%-5% | 454 | A3 | 0.8423 | A3 | 0.7667 |
| size_group | small < 1% | 346 | A1 | 0.7589 | A1 | 0.6649 |

## 主要结论

- 整体测试集最优模型是 A3（U-Net + Boundary Loss），Dice = 0.8075，IoU = 0.7271。
- glioma 仍然是最难分割的肿瘤类型。
- small tumor 仍然比 medium / large tumor 更困难。
- 当前控制变量对比中，Boundary Loss 是最明确有效的改进。
- 当前超参数下，Attention U-Net + Boundary Loss 没有超过 U-Net + Boundary Loss。
- A5 选择 A3 作为后续调参基础模型。
