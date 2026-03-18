import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """多分类 Focal Loss，支持类别权重 alpha。"""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction 必须是 'mean'、'sum' 或 'none'")
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if alpha.dim() != 1:
                raise ValueError("alpha 需要是 1D tensor，长度等于类别数")
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets = targets.long()

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)

        ce_loss = F.nll_loss(log_probs, targets, weight=self.alpha, reduction="none")
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
