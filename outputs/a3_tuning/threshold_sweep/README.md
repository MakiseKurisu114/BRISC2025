# A3 Threshold Sweep

本目录保存 A3 original 和 A3 small oversampling w=3 的阈值扫描结果。

扫描阈值：

```text
0.30 / 0.35 / 0.40 / 0.45 / 0.50
```

目的：

```text
验证 small tumor 是否因为默认 threshold=0.50 太高而被直接抹掉。
```

## A3 Original

| threshold | overall Dice | overall IoU | small Dice | small IoU |
|---:|---:|---:|---:|---:|
| 0.30 | 0.8060 | 0.7251 | 0.7506 | 0.6593 |
| 0.35 | 0.8064 | 0.7256 | 0.7517 | 0.6607 |
| 0.40 | 0.8067 | 0.7261 | 0.7530 | 0.6624 |
| 0.45 | 0.8069 | 0.7263 | 0.7536 | 0.6630 |
| 0.50 | 0.8069 | 0.7263 | 0.7542 | 0.6638 |

## A3 Small Oversampling w=3

| threshold | overall Dice | overall IoU | small Dice | small IoU |
|---:|---:|---:|---:|---:|
| 0.30 | 0.7877 | 0.7006 | 0.7776 | 0.6838 |
| 0.35 | 0.7878 | 0.7010 | 0.7786 | 0.6854 |
| 0.40 | 0.7881 | 0.7014 | 0.7797 | 0.6867 |
| 0.45 | 0.7882 | 0.7018 | 0.7805 | 0.6880 |
| 0.50 | 0.7881 | 0.7018 | 0.7811 | 0.6890 |

## 结论

降低阈值到 0.30-0.45 没有改善 small tumor。两个 checkpoint 都是在默认阈值 0.50 下 small tumor 指标最高。

这说明当前 small tumor 难点不主要是后处理阈值过高导致的小目标被抹掉，更可能来自：

- 模型对小目标定位不够准；
- 小目标置信度分布不够集中；
- 训练采样和 loss 对小目标仍不够平衡；
- small tumor 的形态和边界本身更难。

下一步更建议继续试更温和的 oversampling 权重：

```text
small_sample_weight = 1.5 / 2.0
```

目标是在 small tumor 提升和 overall Dice 之间取得更好的折中。
