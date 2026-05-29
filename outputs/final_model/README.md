# Final Model 总结

本目录汇总最终模型选择、final inference 结果和最终可视化展示。本阶段没有重新训练模型，没有修改数据集文件，也没有复制或修改 `.pt` checkpoint。

最终模型选择读取的已有结果文件：

```text
outputs/a5/summary/overall_test_metrics.csv
outputs/a5/summary/best_by_group.csv
outputs/a3_tuning/final_test.csv
outputs/a3_tuning/final_selection.json
```

选择规则：

1. 先从 `overall_test_metrics.csv` 中选择 A1-A4 主实验里 independent test Dice 最高的模型。
2. 如果 test Dice 并列或非常接近，再比较 test IoU。
3. 再把 `final_test.csv` 中的 A3 tuning final candidate 与主实验最佳模型比较。
4. 只有当 tuning candidate 的 final test Dice / IoU 超过主实验最佳模型时，才替代主实验最佳模型。
5. 不因为 validation 指标更高就直接选择 tuning candidate 作为 final model。

## 候选模型对比

| candidate | role | method | checkpoint | threshold | test Dice | test IoU | selection source |
|---|---|---|---|---:|---:|---:|---|
| A1 | 主实验 | U-Net | `outputs/a1/full/checkpoints/best_unet.pt` | 0.50 | 0.7930 | 0.7078 | `overall_test_metrics.csv` |
| A2 | 主实验 | Attention U-Net | `outputs/a2/full/checkpoints/best_attention_unet.pt` | 0.50 | 0.7937 | 0.7087 | `overall_test_metrics.csv` |
| A3 | 主实验 | U-Net + Boundary Loss | `outputs/a3/full/checkpoints/best_unet_boundary.pt` | 0.50 | 0.8075 | 0.7271 | `overall_test_metrics.csv` |
| A4 | 主实验 | Attention U-Net + Boundary Loss | `outputs/a4/full/checkpoints/best_attention_unet_boundary.pt` | 0.50 | 0.7923 | 0.7093 | `overall_test_metrics.csv` |
| A3_boundary_w03 | supplemental tuning final candidate | U-Net + Boundary Loss, boundary_weight=0.3 | `outputs/a3_tuning/boundary_w03/full/checkpoints/best_unet_boundary.pt` | 0.30 | 0.7986 | 0.7164 | `final_selection.json` + `final_test.csv` |

## 最终选择

最终模型：

```text
experiment = A3
method     = U-Net + Boundary Loss
checkpoint = outputs/a3/full/checkpoints/best_unet_boundary.pt
threshold  = 0.50
test Dice  = 0.8075
test IoU   = 0.7271
```

本目录不复制 `.pt` 文件，只引用原始 checkpoint 路径。

选择 A3 的原因是：A3 是 A1-A4 主实验中 independent test Dice 最高的模型，同时也是 A1-A4 中 test IoU 最高的模型。A3 tuning 是 validation-based supplemental tuning experiment：`A3_boundary_w03 + threshold 0.30` 是根据 validation 指标选出的候选配置，随后只在 test set 上做一次 final test evaluation。其 final test Dice = 0.7986、final test IoU = 0.7164，没有超过原始 A3，因此不替代最终模型。

## Final Inference

final inference 使用：

```text
split      = segmentation_task/test
script     = scripts/run_final_inference.py
checkpoint = outputs/a3/full/checkpoints/best_unet_boundary.pt
threshold  = 0.50
n_samples  = 860
```

本次 final inference 在当前目录下生成：

```text
per_sample_metrics.csv
group_metrics.csv
final_test_summary.csv
selected_examples.csv
figures/
```

`final_test_summary.csv` 中的 final inference 整体结果：

| split | n | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| segmentation_task/test | 860 | 0.8069 | 0.7263 | 0.8221 | 0.8442 |

这里与 A5 选择表中的数值存在很小差异，是因为本次 final inference 汇总的是 `per_sample_metrics.csv` 的逐样本均值；最终模型选择仍以 `final_model.json` 记录的 A5 选择逻辑为准。

## 输出文件

| file | description |
|---|---|
| `final_model.json` | 机器可读的 final model 选择依据、来源文件、checkpoint 和 final inference 输出记录 |
| `final_test_summary.csv` | final model 在 `segmentation_task/test` 上的整体指标 |
| `per_sample_metrics.csv` | 每张测试图的 Dice / IoU / Precision / Recall，以及 tumor / view / size 分组 |
| `group_metrics.csv` | 按 tumor type、view、size group 汇总的平均指标 |
| `selected_examples.csv` | 每张可视化图选中的代表样本 |
| `figures/*.png` | final qualitative examples |

## 可视化结果

所有最终可视化图片保存在 `outputs/final_model/figures/`。

| figure | content |
|---|---|
| `overall_examples.png` | overall 的 2 个 worst、2 个 typical、2 个 best 样本 |
| `tumor_glioma_examples.png` | glioma 的 worst / typical / best 样本 |
| `tumor_meningioma_examples.png` | meningioma 的 worst / typical / best 样本 |
| `tumor_pituitary_examples.png` | pituitary 的 worst / typical / best 样本 |
| `view_axial_examples.png` | axial 的 worst / typical / best 样本 |
| `view_coronal_examples.png` | coronal 的 worst / typical / best 样本 |
| `view_sagittal_examples.png` | sagittal 的 worst / typical / best 样本 |
| `size_small_examples.png` | small tumor 的 worst / typical / best 样本 |
| `size_medium_examples.png` | medium tumor 的 worst / typical / best 样本 |
| `size_large_examples.png` | large tumor 的 worst / typical / best 样本 |

每行依次展示 MRI、ground truth mask、predicted mask 和 contour overlay。overlay 中绿色轮廓表示 ground truth，红色轮廓表示 prediction。

代表样本选择规则：

- best：组内 Dice 最高；
- worst：组内 Dice 最低；
- typical / median：Dice 最接近组内中位数；
- overall：选择 2 个 best、2 个 worst 和 2 个最接近 overall median Dice 的样本；
- 如果某个组样本很少导致重复选择，则去除重复样本。

## 分组观察

根据 `best_by_group.csv`，A3 在 glioma、meningioma、三个 view 和 medium tumor 等多个分组上表现较强。但 small tumor 组的最优模型是 A1 而不是 A3，说明小目标分割仍然是困难点，也说明分组最优模型不一定等于整体最终模型。

本次 final inference 的分组均值：

| group type | group | n | Dice | IoU | 主要观察 |
|---|---|---:|---:|---:|---|
| tumor | glioma | 254 | 0.6619 | 0.5619 | 最困难的 tumor type |
| tumor | meningioma | 306 | 0.9099 | 0.8552 | tumor type 中表现最好 |
| tumor | pituitary | 300 | 0.8244 | 0.7340 | 整体较好，但 small pituitary 仍有失败样本 |
| view | axial | 346 | 0.8015 | 0.7252 | 表现稳定 |
| view | coronal | 257 | 0.8041 | 0.7193 | 表现稳定 |
| view | sagittal | 257 | 0.8168 | 0.7348 | view 组中 Dice 略高 |
| size | small < 1% | 346 | 0.7542 | 0.6638 | 仍然困难，存在多个接近漏检的失败案例 |
| size | medium 1%-5% | 454 | 0.8423 | 0.7667 | size 组中 Dice 最好 |
| size | large > 5% | 60 | 0.8424 | 0.7809 | 平均 IoU 较高，但仍有少数严重失败案例 |

典型成功案例中，绿色 GT 轮廓和红色预测轮廓高度重合，尤其在 meningioma 和 medium tumor 中更明显。失败案例多出现在小病灶或低对比度区域，部分 worst 样本 Dice 接近 0。

机器可读的 final model 选择和 final inference 记录保存在：

```text
outputs/final_model/final_model.json
```
