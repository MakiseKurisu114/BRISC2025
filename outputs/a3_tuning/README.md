# A3 High-Resolution Tuning

本目录保存 A3 的输入分辨率调参结果，不覆盖原始 `outputs/a3/full/`。

调参假设：

```text
image_size=128 时 small tumor 像素太少，提升到 192/256 可能保留更多细节。
```

固定设置：

```text
model           = U-Net
loss            = BCE Loss + Dice Loss + Boundary Loss
boundary_weight = 0.2
base_channels   = 16
epochs          = 20
seed            = 42
```

## 结果对比

| 实验 | image_size | batch_size | test Dice | test IoU | small Dice | small IoU |
|---|---:|---:|---:|---:|---:|---:|
| A3 original | 128 | 8 | 0.8075 | 0.7271 | 0.7542 | 0.6638 |
| A3 image_size 192 | 192 | 8 | 0.7809 | 0.6989 | 0.7447 | 0.6555 |
| A3 image_size 256 | 256 | 8 | 0.7744 | 0.6893 | 0.7414 | 0.6521 |
| A3 small oversampling w=1.5 | 128 | 8 | 0.7826 | 0.7020 | 0.7575 | 0.6699 |
| A3 small oversampling w=2.0 | 128 | 8 | 0.7974 | 0.7153 | 0.7487 | 0.6558 |
| A3 small oversampling w=3 | 128 | 8 | 0.7886 | 0.7024 | 0.7811 | 0.6890 |

## 结论

提高输入分辨率到 192 或 256 没有改善 small tumor，整体测试集 Dice / IoU 也下降。

small tumor oversampling 使用 `WeightedRandomSampler`，将训练集中 small tumor 的期望采样概率从 0.3861 提高到 0.6536。该设置显著提高 small tumor 指标，但整体 test Dice / IoU 下降。

更温和的 oversampling 权重结果：

| small_sample_weight | weighted small probability | test Dice | small Dice | 结论 |
|---:|---:|---:|---:|---|
| 1.5 | 0.4854 | 0.7826 | 0.7575 | small 轻微提升，但整体下降明显 |
| 2.0 | 0.5571 | 0.7974 | 0.7487 | overall 折中最好，但 small 未提升 |
| 3.0 | 0.6536 | 0.7886 | 0.7811 | small 最好，但整体下降 |

阈值扫描显示，在 `threshold=0.30-0.50` 范围内，降低阈值没有改善 small tumor；A3 original 和 oversampling w=3 都是在默认 `threshold=0.50` 下 small tumor 指标最高。

当前结果说明：

- small tumor 表现差不只是由输入缩放到 128 导致；
- 在当前 `epochs=20`、`base_channels=16`、`boundary_weight=0.2` 设置下，高分辨率训练更难收敛；
- small tumor oversampling 是有效方向，但权重选择存在明显 tradeoff；
- 阈值过高不是当前 small tumor 的主要瓶颈；
- `w=2.0` 是本轮 overall 折中最好的 oversampling 设置；
- `w=3.0` 是本轮 small tumor 最好的设置。

## 输出文件

```text
summary.csv
image_size_192/full/
image_size_256_bs8/full/
small_oversampling_w3/full/
small_oversampling_w15/full/
small_oversampling_w2/full/
threshold_sweep/
```
