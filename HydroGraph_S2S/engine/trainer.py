import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from .losses import ExtremeWeightedLoss, HurdleLoss, ConsistencyLoss
from .metrics import S2SMetrics


class S2STrainer:
    """
    Unified Trainer for Spatio-Temporal Graph Neural Networks across all 4 S2S paradigms.
    """
    def __init__(
        self,
        model: nn.Module,
        paradigm: str,
        supports: List[torch.Tensor],
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
        early_stop_patience: int = 15,
        save_dir: str = "./results"
    ):
        self.model = model.to(device)
        self.paradigm = paradigm
        self.supports = [s.to(device) for s in supports]
        self.device = device
        self.grad_clip = grad_clip
        self.patience = early_stop_patience
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Select loss function based on paradigm
        if self.paradigm == "hurdle":
            self.criterion = HurdleLoss(quantiles=[0.50, 0.90, 0.95])
        else:
            self.criterion = ExtremeWeightedLoss()
            
        self.best_val_loss = float("inf")
        self.best_checkpoint_path = os.path.join(self.save_dir, f"best_model_{paradigm}.pt")

    def train_epoch(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            self.optimizer.zero_grad()
            
            x = batch["x"].to(self.device)
            macro = batch["macro"].to(self.device)
            
            if self.paradigm == "pentad":
                y = batch["y"].to(self.device)
                y_pred, _ = self.model(x, macro, self.supports)
                loss = self.criterion(y_pred, y)
                
            elif self.paradigm == "daily_hybrid":
                trend = batch["trend"].to(self.device)
                y = batch["y"].to(self.device)
                y_pred, _, _ = self.model(x, trend, macro, self.supports)
                loss = self.criterion(y_pred, y)
                
            elif self.paradigm == "hurdle":
                y_amt = batch["y_amt"].to(self.device)
                y_occ = batch["y_occ"].to(self.device)
                occ_logits, quantiles_pred, _ = self.model(x, macro, self.supports)
                loss = self.criterion(occ_logits, quantiles_pred, y_occ, y_amt)
                
            elif self.paradigm == "disagg":
                # For disaggregation, monthly total is sum of target days
                y = batch["y"].to(self.device)
                m_total = torch.sum(y, dim=1)
                daily_pred, _, _ = self.model(x, m_total, macro, self.supports)
                loss = self.criterion(daily_pred, y)
                
            else:
                raise ValueError(f"Unknown paradigm: {self.paradigm}")

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = batch["x"].to(self.device)
            macro = batch["macro"].to(self.device)
            
            if self.paradigm == "pentad":
                y = batch["y"].to(self.device)
                y_pred, _ = self.model(x, macro, self.supports)
                loss = self.criterion(y_pred, y)
                eval_pred = y_pred
                eval_target = y
                
            elif self.paradigm == "daily_hybrid":
                trend = batch["trend"].to(self.device)
                y = batch["y"].to(self.device)
                y_pred, _, _ = self.model(x, trend, macro, self.supports)
                loss = self.criterion(y_pred, y)
                eval_pred = y_pred
                eval_target = y
                
            elif self.paradigm == "hurdle":
                y_amt = batch["y_amt"].to(self.device)
                y_occ = batch["y_occ"].to(self.device)
                occ_logits, quantiles_pred, _ = self.model(x, macro, self.supports)
                loss = self.criterion(occ_logits, quantiles_pred, y_occ, y_amt)
                # For hurdle evaluation, expected value = sigmoid(occ_logits) * median quantile (q50)
                prob_occ = torch.sigmoid(occ_logits)
                eval_pred = prob_occ * quantiles_pred[..., 0]  # index 0 is 50% median
                eval_target = y_amt
                
            elif self.paradigm == "disagg":
                y = batch["y"].to(self.device)
                m_total = torch.sum(y, dim=1)
                daily_pred, _, _ = self.model(x, m_total, macro, self.supports)
                loss = self.criterion(daily_pred, y)
                eval_pred = daily_pred
                eval_target = y
                
            total_loss += loss.item()
            num_batches += 1
            all_preds.append(eval_pred.cpu())
            all_targets.append(eval_target.cpu())
            
        avg_loss = total_loss / max(num_batches, 1)
        
        concat_preds = torch.cat(all_preds, dim=0)
        concat_targets = torch.cat(all_targets, dim=0)
        metrics = S2SMetrics.compute_all(concat_preds, concat_targets)
        
        return avg_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        print(f"[{self.paradigm.upper()}] Starting training on {self.device} for {epochs} epochs...")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, max_batches=max_batches)
            val_loss, val_metrics = self.evaluate(val_loader, max_batches=max_batches)
            self.scheduler.step(val_loss)
            
            elapsed = time.time() - t0
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            
            print(
                f"Epoch {epoch:03d}/{epochs:03d} [{elapsed:.1f}s] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val ACC: {val_metrics['ACC']:.4f} | Val RMSE: {val_metrics['RMSE']:.2f}mm | "
                f"Val TS_0.1: {val_metrics['TS_Light_0.1mm']:.3f}"
            )
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[{self.paradigm.upper()}] Early stopping triggered at epoch {epoch}!")
                    break
                    
        # Load best weights
        if os.path.exists(self.best_checkpoint_path):
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))
            
        return history
