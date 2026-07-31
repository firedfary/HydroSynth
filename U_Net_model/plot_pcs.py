import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add project root to path
_curr_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_curr_file))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

def main():
    # 1. Load SST data
    sst_path = config.modelconfig["sst_file"]
    print(f"Loading SST data from: {sst_path}")
    sst = np.load(sst_path)  # [T, n_chan, H, W]
    print(f"SST shape: {sst.shape}")

    # If it is 4D [T, n_chan, H, W], average over the channel dimension to get [T, H, W]
    if sst.ndim == 4:
        sst = np.mean(sst, axis=1)
        print(f"Averaged over channel dim: {sst.shape}")

    T, H, W = sst.shape
    
    # 2. Compute anomalies and reshape
    sst_mean = np.nanmean(sst, axis=0, keepdims=True)
    sst_anom = sst - sst_mean
    X = sst_anom.reshape(T, -1)
    X[~np.isfinite(X)] = 0.0

    # 3. Perform PCA (n_pcs = 3)
    n_pcs = 3
    pca = PCA(n_components=n_pcs, svd_solver='full')
    pcs = pca.fit_transform(X)  # [T, 3]
    
    # Standardize PCs
    pcs = (pcs - pcs.mean(0, keepdims=True)) / (pcs.std(0, keepdims=True) + 1e-8)
    print("PCA completed. Standardized PCs computed.")

    # 4. Generate time list (x-axis)
    # 366 months starting from 1994.01, skipping 2011.09, 2011.10, and 2017.01
    all_months = []
    for year in range(1994, 2025):
        for month in range(1, 13):
            if (year == 2011 and month in [9, 10]) or (year == 2017 and month == 1):
                continue
            if year == 2024 and month > 9:
                break
            all_months.append(year + (month - 1) / 12.0)
    all_months = np.array(all_months)[:T]  # Align to T

    # 5. Plotting with the exact style of the uploaded template (3 vertically stacked subplots)
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 4.5), sharex=True, dpi=300)
    
    # Set transparent figure background
    fig.patch.set_alpha(0.0)
    
    line_color = '#2E659A'  # The blue color matching the template line
    bg_color = '#F0F5FA'    # Light blue-ish background color for each subplot
    border_color = '#333333' # Dark gray border color
    
    for i in range(n_pcs):
        ax = axes[i]
        
        # Set light blue background for the subplot
        ax.set_facecolor(bg_color)
        
        # Plot the PC line
        ax.plot(all_months, pcs[:, i], color=line_color, linewidth=3)
        
        # Set limits
        ax.set_xlim(all_months[0], all_months[-1])
        ax.set_ylim(-3.2, 3.2)
        
        # Configure borders (spines) - all 4 sides visible
        for spine_name in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine_name].set_visible(True)
            ax.spines[spine_name].set_color(border_color)
            ax.spines[spine_name].set_linewidth(2)
            
        # Configure ticks: only x-axis bottom ticks, no labels, no y-axis ticks
        ax.tick_params(
            axis='x', 
            which='both', 
            bottom=True, 
            top=False, 
            labelbottom=False, 
            colors=border_color, 
            direction='out', 
            length=3.5, 
            width=0.8
        )
        ax.tick_params(
            axis='y', 
            which='both', 
            left=False, 
            right=False, 
            labelleft=False
        )

    # Adjust vertical spacing between subplots
    plt.subplots_adjust(hspace=0.25)
    
    # Save as SVG
    output_dir = os.path.join(config.modelconfig["base_data_path"], "picture")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sst_pcs.svg")
    plt.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
    print(f"Successfully saved plot to: {output_path}")

if __name__ == "__main__":
    main()
