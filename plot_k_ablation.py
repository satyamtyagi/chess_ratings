#!/usr/bin/env python3
"""
Generate K-value ablation plot for DirtyGraph in PDF format.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def create_k_ablation_plot(csv_file: str = "dirtygraph_k_detailed_results.csv", 
                          output_pdf: str = "dirtygraph_k_ablation.pdf"):
    """Create K-value ablation plot."""
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Convert 'infinite' to a numeric value for plotting (use 20 for visual purposes)
    df['K_numeric'] = df['K'].apply(lambda x: 20 if x == 'infinite' else int(x))
    df['K_label'] = df['K'].astype(str)
    
    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the line
    ax.plot(df['K_numeric'], df['RMSE'], 
           color='#FF7F0E',  # Orange - DirtyGraph color
           linewidth=3, 
           marker='o', 
           markersize=6,
           markerfacecolor='#FF7F0E',
           markeredgecolor='white',
           markeredgewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Edge Cap K', fontsize=12)
    ax.set_ylabel('RMSE vs true θ', fontsize=12)
    ax.set_title('DirtyGraph: RMSE vs Edge Cap K', fontsize=14, fontweight='bold')
    
    # Set axis limits and ticks
    ax.set_xlim(3, 21)
    ax.set_ylim(0.04, 0.11)
    
    # Set X-axis ticks
    x_ticks = df['K_numeric'].tolist()
    x_labels = [str(k) if k != 'infinite' else '∞' for k in df['K']]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotation for optimal K
    optimal_idx = df['RMSE'].idxmin()
    optimal_k = df.loc[optimal_idx, 'K_numeric']
    optimal_rmse = df.loc[optimal_idx, 'RMSE']
    optimal_k_label = df.loc[optimal_idx, 'K_label']
    
    ax.annotate(f'Optimal: K={optimal_k_label}\nRMSE={optimal_rmse:.6f}', 
                xy=(optimal_k, optimal_rmse), 
                xytext=(optimal_k + 2, optimal_rmse + 0.015),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, 
                ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='red', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"✅ K-ablation plot saved to {output_pdf}")

def create_alternative_k_plot(csv_file: str = "dirtygraph_k_detailed_results.csv", 
                             output_pdf: str = "dirtygraph_k_ablation_log.pdf"):
    """Create alternative plot with log scale to better show the cliff at K=4."""
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Convert 'infinite' to numeric
    df['K_numeric'] = df['K'].apply(lambda x: 20 if x == 'infinite' else int(x))
    df['K_label'] = df['K'].astype(str)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with log scale on Y-axis
    ax.semilogy(df['K_numeric'], df['RMSE'], 
               color='#FF7F0E',  # Orange
               linewidth=3, 
               marker='o', 
               markersize=6,
               markerfacecolor='#FF7F0E',
               markeredgecolor='white',
               markeredgewidth=1)
    
    # Customize
    ax.set_xlabel('Edge Cap K', fontsize=12)
    ax.set_ylabel('RMSE vs true θ (log scale)', fontsize=12)
    ax.set_title('DirtyGraph: RMSE vs Edge Cap K (Log Scale)', fontsize=14, fontweight='bold')
    
    # Set limits and ticks
    ax.set_xlim(3, 21)
    x_ticks = df['K_numeric'].tolist()
    x_labels = [str(k) if k != 'infinite' else '∞' for k in df['K']]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
    
    plt.close()
    print(f"✅ K-ablation log plot saved to {output_pdf}")

def main():
    """Generate both K-ablation plots."""
    csv_file = "dirtygraph_k_detailed_results.csv"
    
    # Create linear scale plot
    create_k_ablation_plot(csv_file, "dirtygraph_k_ablation.pdf")
    
    # Create log scale plot to better show the K=4 cliff
    create_alternative_k_plot(csv_file, "dirtygraph_k_ablation_log.pdf")
    
    print("\n📊 Generated K-ablation plots:")
    print("  1. dirtygraph_k_ablation.pdf - Linear scale with annotation")
    print("  2. dirtygraph_k_ablation_log.pdf - Log scale showing dramatic K=4 cliff")

if __name__ == '__main__':
    main()
