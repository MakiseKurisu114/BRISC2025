# Codex 历史补救摘要：A3 tuning 后续状态

时间：2026-05-29 13:52 CST  
项目路径：`/home/wxy/python_project/kaggle/brisc2025`

## 重要修正：final 选择必须基于 validation set

早期 A3 tuning 中使用 test set 比较 boundary weight、oversampling、Focal Tversky 和 threshold，这些结果现在只保留为 exploratory analysis，不能作为严格 final model/config 选择依据。

已完成规范修正：

```text
val selection 输出：outputs/a3_tuning/summary_val.csv 和 outputs/a3_tuning/threshold_sweep_val.csv
final test 输出：outputs/a3_tuning/final_test.csv
```

严格流程：

```text
train split: 训练已有 checkpoint
val split  : 选择模型设置和 threshold
test split : 只对最终固定配置做一次最终泛化评估
```

Validation 选出的最终配置：

```text
candidate = A3_boundary_w03
checkpoint = outputs/a3_tuning/boundary_w03/full/checkpoints/best_unet_boundary.pt
boundary_weight = 0.3
threshold = 0.30
val Dice = 0.8052
val IoU  = 0.7241
val Precision = 0.8335
val Recall    = 0.8205
```

Final test 结果：

```text
test Dice = 0.7986
test IoU  = 0.7164
test Precision = 0.8235
test Recall    = 0.8311
```

下面旧的一句话结论保留为历史记录，已经不是严格 final 结论。

## 历史一句话结论

路线 A 已完成 A1/A2/A3/A4/A5。后续 A3 tuning 里，新的综合最优模型是：

```text
A3 Boundary w=0.5 + threshold=0.60
test Dice = 0.8122
test IoU  = 0.7324
small Dice = 0.7666
small IoU  = 0.6735
```

small tumor 单项最优仍是：

```text
A3 small oversampling w=3
small Dice = 0.7811
small IoU  = 0.6890
```

但它 overall 明显下降，不适合作为综合主模型。

## 重要操作约束

- 不要重跑 A1/A2/A3/A4/A5，除非用户明确要求。
- 不要上传 `docs/`。
- 后续实验继续放在 `outputs/a3_tuning/`，不要覆盖旧结果。
- 用户强调必须用 GPU。默认沙箱可能会让 `torch.cuda.is_available()` 变成 False；训练/评估需要用提权方式运行，并确认日志里有：

```text
device=cuda
```

- test/train 没有混用。训练/验证只来自 `segmentation_task/train`，测试只来自 `segmentation_task/test`。

## 当前最佳模型和输出

当前推荐主模型：

```text
model = U-Net
loss = BCE Loss + Dice Loss + 0.5 * Boundary Loss
image_size = 128
batch_size = 8
epochs = 20
threshold = 0.60
```

checkpoint：

```text
outputs/a3_tuning/boundary_w05/full/checkpoints/best_unet_boundary.pt
```

完整 threshold=0.60 测试集评估：

```text
outputs/a3_tuning/boundary_w05/full/eval_test_thr060/
```

该目录包含：

```text
metrics.csv
per_sample_metrics.csv
group_metrics.csv
test_examples_worst_best.png
group_visuals/
```

## 已完成的 tuning 实验

### 1. Image size

| 实验 | test Dice | test IoU | small Dice | small IoU |
|---|---:|---:|---:|---:|
| A3 original 128 | 0.8075 | 0.7271 | 0.7542 | 0.6638 |
| image_size 192 | 0.7809 | 0.6989 | 0.7447 | 0.6555 |
| image_size 256 | 0.7744 | 0.6893 | 0.7414 | 0.6521 |

结论：单纯提高输入分辨率没有改善 small tumor。

### 2. Small tumor oversampling

| 实验 | test Dice | test IoU | small Dice | small IoU |
|---|---:|---:|---:|---:|
| oversampling w=1.5 | 0.7826 | 0.7020 | 0.7575 | 0.6699 |
| oversampling w=2.0 | 0.7974 | 0.7153 | 0.7487 | 0.6558 |
| oversampling w=3.0 | 0.7886 | 0.7024 | 0.7811 | 0.6890 |

结论：w=3 small 最好，但 overall 下降，不适合作为综合主模型。

### 3. Focal Tversky

GPU 重训结果：

```text
loss = BCE + Dice + 0.2 * Boundary + 0.2 * Focal Tversky
alpha = 0.3
beta  = 0.7
gamma = 0.75
```

| 实验 | test Dice | test IoU | small Dice | small IoU |
|---|---:|---:|---:|---:|
| Focal Tversky w=0.2 | 0.7975 | 0.7140 | 0.7489 | 0.6544 |

结论：没有提升 small，也没有提升 overall。

### 4. Boundary Loss 权重

| boundary_weight | best val Dice | test Dice | test IoU | small Dice | small IoU |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.7841 | 0.7900 | 0.7062 | 0.7734 | 0.6799 |
| 0.1 | 0.7976 | 0.8017 | 0.7180 | 0.7525 | 0.6596 |
| 0.2 original | 0.8043 | 0.8075 | 0.7271 | 0.7542 | 0.6638 |
| 0.3 | 0.8053 | 0.7991 | 0.7172 | 0.7506 | 0.6574 |
| 0.5 | 0.8014 | 0.8120 | 0.7319 | 0.7644 | 0.6704 |

结论：

```text
overall 最优：Boundary w=0.5
small 单项较好但 overall 差：Boundary w=0.05
综合推荐：Boundary w=0.5
```

### 5. Threshold sweep

旧 sweep：

- A3 original：0.50 最好。
- oversampling w=3：0.50 最好。

新 sweep：

```text
checkpoint = outputs/a3_tuning/boundary_w05/full/checkpoints/best_unet_boundary.pt
thresholds = 0.30 / 0.35 / 0.40 / 0.45 / 0.50 / 0.55 / 0.60
```

| threshold | overall Dice | overall IoU | small Dice | small IoU |
|---:|---:|---:|---:|---:|
| 0.30 | 0.8107 | 0.7298 | 0.7596 | 0.6641 |
| 0.35 | 0.8111 | 0.7304 | 0.7613 | 0.6662 |
| 0.40 | 0.8114 | 0.7310 | 0.7627 | 0.6680 |
| 0.45 | 0.8115 | 0.7312 | 0.7636 | 0.6693 |
| 0.50 | 0.8116 | 0.7314 | 0.7644 | 0.6704 |
| 0.55 | 0.8117 | 0.7317 | 0.7655 | 0.6719 |
| 0.60 | 0.8118 | 0.7320 | 0.7666 | 0.6735 |

完整 `threshold=0.60` eval：

```text
outputs/a3_tuning/boundary_w05/full/eval_test_thr060/
test Dice = 0.8122
test IoU  = 0.7324
```

结论：0.60 有轻微收益，但主要收益来自 Boundary w=0.5 本身。

## 后处理连通域检查

当前代码没有做小连通域删除。

评估逻辑是：

```text
prob = sigmoid(logits)
pred = prob > threshold
```

没有 connected components、remove_small_objects、开闭运算等后处理。

建议报告里写：

```text
当前不使用小连通域删除，因为 small tumor 是主要短板，删除小连通域有误删真实病灶风险。
```

## 最新修改过的脚本

新增：

```text
scripts/train_a3_unet_focal_tversky.py
```

修改：

```text
src/losses.py
scripts/evaluate_checkpoint.py
scripts/generate_group_visuals.py
```

`evaluate_checkpoint.py` 和 `generate_group_visuals.py` 已支持：

```text
--threshold
```

用于生成完整 threshold=0.60 的 metrics 和可视化。

## 历史下一步建议已作废

下面关于 `boundary_w05 + threshold=0.60 + TTA` 的建议是 test-based exploratory 阶段的历史建议，已经不符合严格 final 选参流程。若后续要做 TTA，必须先在 validation set 上比较 TTA / non-TTA 和 threshold，再只对 val 选出的最终配置做一次 test。

历史建议原文：

```text
A3 Boundary w=0.5 + threshold=0.60 + TTA
```

TTA 先做保守版本：

```text
original
horizontal flip
vertical flip
horizontal + vertical flip
```

流程：

1. 对每张 test image 做 4 个版本预测。
2. 翻转预测再翻回原方向。
3. 平均 probability map。
4. 用 threshold=0.60 二值化。
5. 输出完整 eval 到：

```text
outputs/a3_tuning/boundary_w05/full/eval_test_thr060_tta/
```

原因：

- 不需要重新训练。
- 不改变 train/test 划分。
- 风险低。
- 可能进一步稳定边界和 small tumor。

如果 TTA 没提升，也可以写入报告：TTA 对当前 BRISC 2D U-Net 设定收益有限。

## 下个对话可复制的提示词

```text
这是 BRISC 2025 脑肿瘤 MRI 分割项目，路径是 /home/wxy/python_project/kaggle/brisc2025。请先读 docs/CODEX_HISTORY_RESCUE_019e6e35.md、docs/CODEX_HISTORY_RESCUE_019e6f3d.md 和 docs/CODEX_HISTORY_RESCUE_019e6f_next.md。

当前路线 A 已完成 A1/A2/A3/A4/A5。A3 tuning 已完成 image size、small oversampling、threshold sweep、Focal Tversky、Boundary Loss 权重重调。不要重跑 A1/A2/A3/A4/A5。不要上传 docs/。

重要修正：A3 tuning 早期 test-based 结果只作为 exploratory analysis，不能作为 final 选参依据。现在已完成规范 final 流程：
val selection = outputs/a3_tuning/summary_val.csv / outputs/a3_tuning/threshold_sweep_val.csv
final test = outputs/a3_tuning/final_test.csv

Validation set 选出的 final config：
candidate = A3_boundary_w03
checkpoint = outputs/a3_tuning/boundary_w03/full/checkpoints/best_unet_boundary.pt
boundary_weight = 0.3
threshold = 0.30
val Dice = 0.8052
val IoU = 0.7241

Final test 只对该固定配置评估一次：
test Dice = 0.7986
test IoU = 0.7164
test Precision = 0.8235
test Recall = 0.8311

训练/评估必须使用 GPU，运行日志必须确认 device=cuda。后续如果继续实验，仍然必须先用 val 选择配置，再只用 test 做最终一次汇报。
```
