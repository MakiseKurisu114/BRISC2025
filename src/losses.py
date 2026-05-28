import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + (1.0 - self.bce_weight) * self.dice(logits, targets)


class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth = smooth

    def _boundary(self, mask: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=pad)
        eroded = -F.max_pool2d(-mask, kernel_size=self.kernel_size, stride=1, padding=pad)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pred_boundary = self._boundary(probs)
        target_boundary = self._boundary(targets)

        pred_boundary = pred_boundary.flatten(start_dim=1)
        target_boundary = target_boundary.flatten(start_dim=1)
        intersection = (pred_boundary * target_boundary).sum(dim=1)
        denominator = pred_boundary.sum(dim=1) + target_boundary.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceBoundaryLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, boundary_weight: float = 0.2) -> None:
        super().__init__()
        self.region_loss = BCEDiceLoss(bce_weight=bce_weight)
        self.boundary_loss = BoundaryLoss()
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.region_loss(logits, targets) + self.boundary_weight * self.boundary_loss(logits, targets)
