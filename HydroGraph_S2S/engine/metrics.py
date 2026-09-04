import numpy as np
import torch
from typing import Dict, List, Union


class S2SMetrics:
    """
    Comprehensive Meteorological and Hydrological Evaluation Metrics for Precipitation Forecasting:
    - Anomaly Correlation Coefficient (ACC)
    - Root Mean Square Error (RMSE) & Mean Absolute Error (MAE)
    - Kling-Gupta Efficiency (KGE)
    - Threat Score (TS / CSI), Equitable Threat Score (ETS), POD, FAR, BIAS across multiple rain grades
    """
    THRESHOLDS = {
        "Light_0.1mm": 0.1,
        "Moderate_10mm": 10.0,
        "Heavy_25mm": 25.0,
        "Storm_50mm": 50.0
    }

    @staticmethod
    def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    @classmethod
    def compute_all(
        cls,
        preds: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        climatology: Union[torch.Tensor, np.ndarray, None] = None
    ) -> Dict[str, float]:
        """
        preds: (B, T, N) or (T, N)
        targets: (B, T, N) or (T, N)
        """
        y_pred = cls.to_numpy(preds).flatten()
        y_true = cls.to_numpy(targets).flatten()
        
        # Clean any remaining NaNs
        valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        
        metrics = {}
        
        # 1. Continuous Error Metrics
        metrics["RMSE"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        metrics["MAE"] = float(np.mean(np.abs(y_pred - y_true)))
        
        # Correlation
        if np.std(y_pred) > 1e-6 and np.std(y_true) > 1e-6:
            r = float(np.corrcoef(y_pred, y_true)[0, 1])
            metrics["Corr"] = r
        else:
            metrics["Corr"] = 0.0

        # ACC (Anomaly Correlation Coefficient)
        if climatology is not None:
            clim = cls.to_numpy(climatology).flatten()[valid_mask]
            anom_pred = y_pred - clim
            anom_true = y_true - clim
            if np.std(anom_pred) > 1e-6 and np.std(anom_true) > 1e-6:
                metrics["ACC"] = float(np.corrcoef(anom_pred, anom_true)[0, 1])
            else:
                metrics["ACC"] = 0.0
        else:
            # Fallback ACC to centered correlation
            metrics["ACC"] = metrics["Corr"]

        # 2. Kling-Gupta Efficiency (KGE)
        mu_p = np.mean(y_pred)
        mu_t = np.mean(y_true)
        std_p = np.std(y_pred)
        std_t = np.std(y_true)
        
        if std_t > 1e-6 and mu_t > 1e-6:
            r = metrics["Corr"]
            alpha = std_p / std_t
            beta = mu_p / mu_t
            kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
            metrics["KGE"] = float(kge)
        else:
            metrics["KGE"] = 0.0

        # 3. Categorical Contingency Metrics (TS, ETS, POD, FAR)
        for grade_name, th in cls.THRESHOLDS.items():
            pred_bin = (y_pred >= th)
            true_bin = (y_true >= th)
            
            tp = float(np.sum(pred_bin & true_bin))
            fp = float(np.sum(pred_bin & ~true_bin))
            fn = float(np.sum(~pred_bin & true_bin))
            tn = float(np.sum(~pred_bin & ~true_bin))
            total = tp + fp + fn + tn
            
            # Threat Score (TS / CSI)
            ts_denom = tp + fp + fn
            metrics[f"TS_{grade_name}"] = float(tp / ts_denom) if ts_denom > 0 else 0.0
            
            # Equitable Threat Score (ETS)
            dr = ((tp + fp) * (tp + fn)) / total if total > 0 else 0.0
            ets_denom = tp + fp + fn - dr
            metrics[f"ETS_{grade_name}"] = float((tp - dr) / ets_denom) if ets_denom > 0 else 0.0
            
            # Probability of Detection (POD)
            metrics[f"POD_{grade_name}"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            # False Alarm Rate (FAR)
            metrics[f"FAR_{grade_name}"] = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
            
        return metrics
