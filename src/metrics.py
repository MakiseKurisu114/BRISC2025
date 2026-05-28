import torch


def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    preds = preds.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    union = pred_sum + target_sum - intersection

    eps = 1e-7
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()
