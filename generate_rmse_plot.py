#!/usr/bin/env python3
"""
Generate RMSE plot in PDF format matching the provided design.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def create_rmse_plot(csv_file: str, output_pdf: str = "synthetic_stream_rmse.pdf"):
    """Create RMSE plot matching the provided design."""
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Set up the plot style to match the image
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors to match the image
    colors = {
        'DirtyGraph (K=10)': '#FF7F0E',           # Orange
        'Batch BT (MM)': '#1F77B4',              # Light blue  
        'OPDN-5 (counts diag-Newton)': '#2CA02C', # Green
        'BT--SGD (duels; 15 passes)': '#FFFF00',  # Yellow
        'Diag-Newton (streaming)': '#0000FF',     # Dark blue (different from Batch BT)
        'FTRL-Prox': '#FF0000',                   # Red
        'ISGD': '#9467BD'                         # Purple
    }
    
    # Get unique algorithms
    algorithms = df['Algorithm'].unique()
    
    # Since we have shuffle data instead of trajectory data, 
    # we'll simulate a trajectory using the shuffle runs as x-axis points
    # and show the RMSE variation across shuffles
    
    # Use actual shuffle numbers as X-axis (1 to 10)
    shuffle_numbers = np.arange(1, 11)  # 1, 2, 3, ..., 10
    
    # Plot each algorithm
    for algorithm in algorithms:
        if algorithm in colors:
            # Get RMSE values for this algorithm across shuffles
            alg_data = df[df['Algorithm'] == algorithm].sort_values('Shuffle_Run')
            rmse_values = alg_data['RMSE'].values
            
            # Plot the line
            ax.plot(shuffle_numbers, rmse_values, 
                   color=colors[algorithm], 
                   linewidth=2, 
                   marker='o', 
                   markersize=4,
                   label=algorithm)
    
    # Customize the plot for shuffle robustness analysis
    ax.set_xlabel('Shuffle Number', fontsize=12)
    ax.set_ylabel('RMSE vs true θ', fontsize=12)
    ax.set_title('Shuffle Robustness: RMSE vs Shuffle Number', fontsize=14, fontweight='bold')
    
    # Set axis limits and ticks for shuffle numbers
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.03, 0.35)
    
    # Set X-axis ticks to show shuffle numbers 1-10
    ax.set_xticks(shuffle_numbers)
    ax.set_xticklabels(shuffle_numbers)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Remove top and right spines (border lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend inside plot area (upper center-right where there's free space)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, 
             bbox_to_anchor=(0.65, 0.98), fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"✅ RMSE plot saved to {output_pdf}")

def create_alternative_plot(csv_file: str, output_pdf: str = "rmse_shuffle_robustness.pdf"):
    """Create an alternative plot showing shuffle robustness."""
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors
    colors = {
        'DirtyGraph (K=10)': '#FF7F0E',
        'Batch BT (MM)': '#1F77B4',
        'OPDN-5 (counts diag-Newton)': '#2CA02C',
        'BT--SGD (duels; 15 passes)': '#FFFF00',
        'Diag-Newton (streaming)': '#0000FF',
        'FTRL-Prox': '#FF0000',
        'ISGD': '#9467BD'
    }
    
    # Create box plot data
    algorithms = []
    rmse_data = []
    
    for algorithm in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == algorithm]
        algorithms.append(algorithm)
        rmse_data.append(alg_data['RMSE'].values)
    
    # Create box plot
    box_plot = ax.boxplot(rmse_data, labels=algorithms, patch_artist=True, widths=0.6)
    
    # Color the boxes
    for patch, algorithm in zip(box_plot['boxes'], algorithms):
        if algorithm in colors:
            patch.set_facecolor(colors[algorithm])
            patch.set_alpha(0.7)
    
    # Customize
    ax.set_ylabel('RMSE vs true θ', fontsize=12)
    ax.set_title('Shuffle Robustness: RMSE Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"✅ Alternative RMSE plot saved to {output_pdf}")

def main():
    """Generate both plot styles."""
    csv_file = "shuffle_corrected_renamed.csv"
    
    # Create trajectory-style plot (matching the image)
    create_rmse_plot(csv_file, "synthetic_stream_rmse.pdf")
    
    # Create robustness box plot as alternative
    create_alternative_plot(csv_file, "rmse_shuffle_robustness.pdf")
    
    print("\n📊 Generated two PDF plots:")
    print("  1. synthetic_stream_rmse.pdf - Trajectory style (matches your image)")
    print("  2. rmse_shuffle_robustness.pdf - Box plot showing shuffle variance")

if __name__ == '__main__':
    main()
