import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ExtremeWeightedLoss(nn.Module):
    """
    Weighted Huber / L1 Loss that assigns progressively higher penalty to extreme precipitation events:
    - [0, 0.1): weight = 0.5 (suppress tiny drizzle noise)
    - [0.1, 10): weight = 1.0 (light rain)
    - [10, 25): weight = 2.5 (moderate rain)
    - [25, 50): weight = 5.0 (heavy rain)
    - [50, inf): weight = 10.0 (torrential rain / extreme storms)
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.abs(pred - target)
        huber_loss = torch.where(
            error < self.delta,
            0.5 * (error ** 2),
            self.delta * (error - 0.5 * self.delta)
        )
        
        # Calculate dynamic meteorological weights based on target intensity
        weights = torch.ones_like(target)
        weights = torch.where(target < 0.1, torch.full_like(weights, 0.5), weights)
        weights = torch.where((target >= 10.0) & (target < 25.0), torch.full_like(weights, 2.5), weights)
        weights = torch.where((target >= 25.0) & (target < 50.0), torch.full_like(weights, 5.0), weights)
        weights = torch.where(target >= 50.0, torch.full_like(weights, 10.0), weights)
        
        weighted_loss = weights * huber_loss
        return torch.mean(weighted_loss)


class QuantileLoss(nn.Module):
    """
    Asymmetric Pinball Loss for multi-quantile precipitation regression:
    L_q(y, y_hat) = max(q * (y - y_hat), (1 - q) * (y_hat - y))
    """
    def __init__(self, quantiles: List[float] = [0.50, 0.90, 0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: (B, T, N, num_quantiles)
        target: (B, T, N)
        """
        target_expanded = target.unsqueeze(-1)  # (B, T, N, 1)
        errors = target_expanded - preds        # (B, T, N, num_quantiles)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            err = errors[..., i]
            loss_q = torch.max(q * err, (q - 1.0) * err)
            losses.append(torch.mean(loss_q))
            
        return sum(losses) / len(losses)


class HurdleLoss(nn.Module):
    """
    Combined Two-Stage Hurdle Loss:
    Loss = lambda_occ * BCE(occ, occ_hat) + lambda_amt * QuantileLoss(y, y_hat)
    """
    def __init__(
        self,
        quantiles: List[float] = [0.50, 0.90, 0.95],
        lambda_occ: float = 1.0,
        lambda_amt: float = 1.0
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.quantile_loss = QuantileLoss(quantiles)
        self.lambda_occ = lambda_occ
        self.lambda_amt = lambda_amt

    def forward(
        self,
        occ_logits: torch.Tensor,
        quantiles_pred: torch.Tensor,
        y_occ: torch.Tensor,
        y_amt: torch.Tensor
    ) -> torch.Tensor:
        loss_occ = self.bce(occ_logits, y_occ)
        loss_amt = self.quantile_loss(quantiles_pred, y_amt)
        return self.lambda_occ * loss_occ + self.lambda_amt * loss_amt


class ConsistencyLoss(nn.Module):
    """
    Physical Consistency Regularization:
    Penalizes the divergence between summed daily predictions and monthly dynamical model target:
    Loss = || sum_{t=1}^{30} y_daily(t, i) - y_monthly(i) ||_2^2
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, daily_preds: torch.Tensor, monthly_target: torch.Tensor) -> torch.Tensor:
        """
        daily_preds: (B, T, N)
        monthly_target: (B, N)
        """
        daily_sum = torch.sum(daily_preds, dim=1)  # (B, N)
        diff = daily_sum - monthly_target
        loss = torch.mean(diff ** 2)
        return self.weight * loss
