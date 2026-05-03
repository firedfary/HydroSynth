# XGBoost Monthly Downscaling Forecast

This project implements a monthly precipitation forecasting model using XGBoost, adapted from the yearly forecasting method in `D:\XGBoost-Downscaling-Model`.

## Data Sources
- **Climate Model Data**: `D:\MODESv21_ecmwf_seas51` (SEAS5 monthly em)
- **Sea Surface Temperature**: `D:\ersst_data` (ERSST v5)
- **Observed Precipitation**: `E:\HydroSynth\utils\observe_data24.csv`

## Methodology
1. **EOF Analysis**: Extract the top 20 Principal Components (PCs) for:
   - Observed precipitation anomaly (target)
   - Climate model variables (TP, SLP, H500, SST) at Lead-1.
   - Observational SST (ERSST) from the previous month.
2. **XGBoost Regression**: For each month (1-12), an XGBoost model is trained to predict the observed precip PCs from the concatenated model/SST PCs.
3. **Validation**: Leave-One-Year-Out Cross-Validation (LOOCV) is used to evaluate the model's accuracy (ACC).
4. **Reconstruction**: The predicted PCs are used to reconstruct the spatial precipitation anomaly field.

## Usage
Run the main script using the `xgb` conda environment:
```powershell
conda run -n xgb python train_monthly.py
```

## Results
Results including ACC scores and spatial predictions are saved in the `results/` directory.
