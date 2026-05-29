# Final Model Summary

This directory summarizes the final model selection and the final inference / visualization run. No training was rerun, no dataset file was modified, and no checkpoint file was copied or changed.

Source files:

```text
outputs/a5/summary/overall_test_metrics.csv
outputs/a5/summary/best_by_group.csv
outputs/a3_tuning/final_test.csv
outputs/a3_tuning/final_selection.json
```

Selection rule:

1. Select the best A1-A4 main experiment by independent test Dice from `overall_test_metrics.csv`.
2. If test Dice is tied or very close, compare test IoU.
3. Compare the A3 tuning final candidate from `final_test.csv` against the best main experiment.
4. Only replace the main-experiment winner if the tuning candidate has higher final test Dice / IoU.
5. Do not use validation metrics alone to select the final model.

## Candidate Comparison

| candidate | role | method | checkpoint | threshold | test Dice | test IoU | selection source |
|---|---|---|---|---:|---:|---:|---|
| A1 | main experiment | U-Net | `outputs/a1/full/checkpoints/best_unet.pt` | 0.50 | 0.7930 | 0.7078 | `overall_test_metrics.csv` |
| A2 | main experiment | Attention U-Net | `outputs/a2/full/checkpoints/best_attention_unet.pt` | 0.50 | 0.7937 | 0.7087 | `overall_test_metrics.csv` |
| A3 | main experiment | U-Net + Boundary Loss | `outputs/a3/full/checkpoints/best_unet_boundary.pt` | 0.50 | 0.8075 | 0.7271 | `overall_test_metrics.csv` |
| A4 | main experiment | Attention U-Net + Boundary Loss | `outputs/a4/full/checkpoints/best_attention_unet_boundary.pt` | 0.50 | 0.7923 | 0.7093 | `overall_test_metrics.csv` |
| A3_boundary_w03 | supplemental tuning final candidate | U-Net + Boundary Loss, boundary_weight=0.3 | `outputs/a3_tuning/boundary_w03/full/checkpoints/best_unet_boundary.pt` | 0.30 | 0.7986 | 0.7164 | `final_selection.json` + `final_test.csv` |

## Final Selection

Final model:

```text
experiment = A3
method     = U-Net + Boundary Loss
checkpoint = outputs/a3/full/checkpoints/best_unet_boundary.pt
threshold  = 0.50
test Dice  = 0.8075
test IoU   = 0.7271
```

The `.pt` file is not copied into this directory. The final model summary only references the original checkpoint path.

A3 is selected because it is the best A1-A4 main experiment on the independent test set by test Dice, and it also has the highest test IoU among A1-A4. A3 tuning is a validation-based supplemental tuning experiment: `A3_boundary_w03 + threshold 0.30` was selected from validation metrics, then evaluated once on the test set. Its final test Dice = 0.7986 and final test IoU = 0.7164 do not exceed original A3, so it does not replace the final model.

## Final Inference

Final inference was run on:

```text
split      = segmentation_task/test
script     = scripts/run_final_inference.py
checkpoint = outputs/a3/full/checkpoints/best_unet_boundary.pt
threshold  = 0.50
n_samples  = 860
```

New final inference outputs saved in this directory:

```text
per_sample_metrics.csv
group_metrics.csv
final_test_summary.csv
selected_examples.csv
figures/
```

Final inference summary from `final_test_summary.csv`:

| split | n | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| segmentation_task/test | 860 | 0.8069 | 0.7263 | 0.8221 | 0.8442 |

The small difference from the A5 selection table is due to this final run reporting the mean of `per_sample_metrics.csv`; final model selection still follows `final_model.json` and the A5 summary rule.

## Output Files

| file | description |
|---|---|
| `final_model.json` | machine-readable final model selection, source files, selected checkpoint, and final inference outputs |
| `final_test_summary.csv` | overall final inference metrics on `segmentation_task/test` |
| `per_sample_metrics.csv` | per-image Dice / IoU / Precision / Recall with tumor, view, and size group |
| `group_metrics.csv` | mean metrics by tumor type, imaging view, and size group |
| `selected_examples.csv` | samples selected for each visualization figure |
| `figures/*.png` | final qualitative examples |

## Visualizations

All qualitative figures are saved under `outputs/final_model/figures/`.

| figure | content |
|---|---|
| `overall_examples.png` | 2 worst, 2 typical, and 2 best overall examples |
| `tumor_glioma_examples.png` | glioma worst / typical / best examples |
| `tumor_meningioma_examples.png` | meningioma worst / typical / best examples |
| `tumor_pituitary_examples.png` | pituitary worst / typical / best examples |
| `view_axial_examples.png` | axial worst / typical / best examples |
| `view_coronal_examples.png` | coronal worst / typical / best examples |
| `view_sagittal_examples.png` | sagittal worst / typical / best examples |
| `size_small_examples.png` | small tumor worst / typical / best examples |
| `size_medium_examples.png` | medium tumor worst / typical / best examples |
| `size_large_examples.png` | large tumor worst / typical / best examples |

Each row shows MRI, ground truth mask, predicted mask, and an overlay. In overlays, ground truth contours are green and predicted contours are red.

Representative sample selection:

- best: highest Dice in the group;
- worst: lowest Dice in the group;
- typical / median: Dice closest to the group median;
- overall: two best, two worst, and two samples closest to the overall median Dice;
- if a group has duplicate selections because of a very small sample count, duplicates are removed.

## Group Notes

From `best_by_group.csv`, A3 is strongest across many groups, including glioma, meningioma, all three imaging views, and medium tumors. However, the small tumor group is best with A1 rather than A3, showing that small tumor segmentation remains a key difficulty and that the group-wise best model is not always the overall final model.

Final inference group means:

| group type | group | n | Dice | IoU | main observation |
|---|---|---:|---:|---:|---|
| tumor | glioma | 254 | 0.6619 | 0.5619 | hardest tumor type |
| tumor | meningioma | 306 | 0.9099 | 0.8552 | strongest tumor-type performance |
| tumor | pituitary | 300 | 0.8244 | 0.7340 | good overall, but small pituitary failures remain |
| view | axial | 346 | 0.8015 | 0.7252 | stable |
| view | coronal | 257 | 0.8041 | 0.7193 | stable |
| view | sagittal | 257 | 0.8168 | 0.7348 | slightly highest view Dice |
| size | small < 1% | 346 | 0.7542 | 0.6638 | still difficult, with multiple near-miss failures |
| size | medium 1%-5% | 454 | 0.8423 | 0.7667 | best size group by Dice |
| size | large > 5% | 60 | 0.8424 | 0.7809 | high IoU on average, but rare severe failures exist |

Typical success cases have close overlap between green and red contours, especially for meningioma and medium-size tumors. Failure cases often show missed or weak predictions for subtle small lesions, and several worst examples have Dice near zero.

Machine-readable selection details are saved in:

```text
outputs/final_model/final_model.json
```
