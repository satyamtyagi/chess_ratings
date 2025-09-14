#!/usr/bin/env python3
"""
RMSE Visualization Script

This script creates visualizations from the RMSE results calculated for different
rating algorithms across shuffled datasets.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter


def load_rmse_data(file_path):
    """Load RMSE data from CSV file."""
    return pd.read_csv(file_path)


def create_box_plot(df, output_file):
    """Create box plot of RMSE values for each algorithm."""
    plt.figure(figsize=(10, 6))
    
    # Exclude the 'shuffle' column and melt the data
    melt_df = df.drop('shuffle', axis=1).melt(var_name='Algorithm', value_name='RMSE')
    
    # Create box plot
    sns.boxplot(x='Algorithm', y='RMSE', data=melt_df)
    
    # Add title and labels
    plt.title('RMSE Distribution Across Algorithms (vs. Bradley-Terry)', fontsize=14)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('RMSE (Root Mean Square Error)', fontsize=12)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Add statistics as text annotations
    stats = {}
    for algo in df.columns[1:]:  # Skip 'shuffle' column
        stats[algo] = {
            'mean': df[algo].mean(),
            'std': df[algo].std(),
            'min': df[algo].min(),
            'max': df[algo].max()
        }
    
    # Display statistics
    textstr = '\n'.join([
        f"{algo}: {s['mean']:.6f} ± {s['std']:.6f}" 
        for algo, s in stats.items()
    ])
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.figtext(0.15, 0.02, textstr, fontsize=9, bbox=props)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    print(f"Box plot saved to {output_file}")
    
    # Close the figure
    plt.close()


def create_line_plot(df, output_file):
    """Create line plot showing RMSE values for each shuffle."""
    plt.figure(figsize=(10, 6))
    
    # Create line plot for each algorithm
    for algo in df.columns[1:]:  # Skip 'shuffle' column
        plt.plot(df['shuffle'], df[algo], marker='o', label=algo)
    
    # Add title and labels
    plt.title('RMSE by Shuffle Number for Each Algorithm', fontsize=14)
    plt.xlabel('Shuffle Number', fontsize=12)
    plt.ylabel('RMSE (Root Mean Square Error)', fontsize=12)
    
    # Add legend
    plt.legend()
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis to integer values
    plt.xticks(df['shuffle'])
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    print(f"Line plot saved to {output_file}")
    
    # Close the figure
    plt.close()


def create_log_line_plot(df, output_file):
    """Create logarithmic scale line plot showing RMSE values for each shuffle."""
    plt.figure(figsize=(11, 6))  # Slightly wider figure
    
    # Create line plot for each algorithm with log scale
    for algo in df.columns[1:]:  # Skip 'shuffle' column
        plt.semilogy(df['shuffle'], df[algo], marker='o', label=algo)
    
    # Add title and labels
    plt.title('RMSE by Shuffle Number (Log Scale)', fontsize=14)
    plt.xlabel('Shuffle Number', fontsize=12)
    plt.ylabel('RMSE (Log Scale)', fontsize=12)
    
    # Format y-axis with fewer ticks and scientific notation
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
    
    # Add legend with smaller font size and moved to right side
    plt.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add grid lines (with special settings for log scale)
    plt.grid(True, which="both", ls="-", alpha=0.7)
    
    # Set x-axis to integer values
    plt.xticks(df['shuffle'])
    
    # Improve layout with more padding on right for legend
    plt.tight_layout(pad=2.0, rect=[0, 0, 0.85, 1])
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    print(f"Log-scale line plot saved to {output_file}")
    
    # Close the figure
    plt.close()


def create_broken_axis_plot(df, output_file):
    """Create broken axis plot to better show the difference between algorithms."""
    # Create figure with a broken y-axis
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)  # Increased spacing between subplots
    
    # Create two subplots with different scales
    ax1 = plt.subplot(gs[0])  # Top plot for others
    ax2 = plt.subplot(gs[1])  # Bottom plot for DG (flipped from previous version)
    
    # Use fixed break points as requested
    top_min = 0.0060    # Top axis starts at this value
    bottom_max = 0.0020  # Bottom axis ends at this value
    
    # Plot other algorithms in top subplot
    colors = ['C1', 'C2', 'C3']
    i = 0
    for algo in df.columns[1:]:  # Skip 'shuffle' column
        if algo != 'DG':
            ax1.plot(df['shuffle'], df[algo], marker='o', label=algo, color=colors[i])
            i += 1
    
    # Set y-limit for top plot with fixed minimum value
    ax1.set_ylim(top_min, None)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))  # Limit number of ticks
    
    # Plot DG in bottom subplot
    ax2.plot(df['shuffle'], df['DG'], marker='o', label='DG', color='C0')
    ax2.set_ylim(0, bottom_max)  # Set y-limit for bottom plot with fixed maximum
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))  # Limit number of ticks
    
    # Indicate the broken axis
    d = .015  # Size of diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # Bottom-left diagonal
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)  # Bottom-right diagonal
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)  # Top-left diagonal
    ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)  # Top-right diagonal
    
    # Format y-axis to show values clearly
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    
    # Make sure 0.0060 and 0.0020 are included in the y-axis ticks
    ax1.set_yticks([0.0060] + list(ax1.get_yticks()[1:]))
    ax2.set_yticks(list(ax2.get_yticks()[:-1]) + [0.0020])
    
    # Add title and labels
    fig.suptitle('RMSE by Shuffle Number (Broken Axis)', fontsize=14)
    ax2.set_xlabel('Shuffle Number', fontsize=12)
    fig.text(0.04, 0.5, 'RMSE (Root Mean Square Error)', va='center', rotation='vertical', fontsize=12)
    
    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis to integer values
    ax1.set_xticks(df['shuffle'])
    ax2.set_xticks(df['shuffle'])
    
    # Only show x labels on bottom plot
    ax1.set_xticklabels([])
    
    # Create combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    
    # Improve layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)  # Reduce space between subplots
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    print(f"Broken-axis plot saved to {output_file}")
    
    # Close the figure
    plt.close()


def create_relative_plot(df, output_file):
    """Create relative performance plot showing how much worse each algorithm is compared to the best."""
    plt.figure(figsize=(11, 6))  # Slightly wider figure
    
    # Calculate relative performance (DG is baseline)
    baseline = df['DG'].mean()
    relative_df = df.copy()
    
    for algo in df.columns[1:]:  # Skip 'shuffle' column
        if algo != 'DG':
            relative_df[algo] = df[algo] / baseline
    
    relative_df['DG'] = 1.0  # DG is baseline (1.0x)
    
    # Create line plot for each algorithm
    for algo in relative_df.columns[1:]:  # Skip 'shuffle' column
        plt.plot(relative_df['shuffle'], relative_df[algo], marker='o', label=algo)
    
    # Add title and labels
    plt.title('Relative Performance vs DG (Lower is Better)', fontsize=14)
    plt.xlabel('Shuffle Number', fontsize=12)
    plt.ylabel('RMSE Ratio (compared to DG)', fontsize=12)
    
    # Add horizontal line at y=1 (DG baseline)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Add annotation for the baseline
    plt.text(1.1, 1.05, 'DG baseline (1.0)', fontsize=10)
    
    # Add legend with smaller font and moved to right side for better spacing
    plt.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis to integer values
    plt.xticks(df['shuffle'])
    
    # Format y-axis with fewer ticks to avoid overlap
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(6))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    
    # Add text annotations for ratio values with clearer formatting
    mean_ratios = {algo: relative_df[algo].mean() for algo in relative_df.columns if algo != 'shuffle'}
    textstr = '\n'.join([f"{algo}: {ratio:.0f}x worse than DG" for algo, ratio in mean_ratios.items() if algo != 'DG'])
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.figtext(0.15, 0.02, textstr, fontsize=10, bbox=props)
    
    # Improve layout with more padding on right for legend
    plt.tight_layout(pad=2.0, rect=[0, 0, 0.85, 1])
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    print(f"Relative performance plot saved to {output_file}")
    
    # Close the figure
    plt.close()


def create_heat_map(df, output_file):
    """Create heat map of RMSE values."""
    plt.figure(figsize=(12, 8))
    
    # Create a pivot table for the heat map
    heat_df = df.set_index('shuffle')
    
    # Create heat map
    sns.heatmap(heat_df, annot=True, cmap='viridis', fmt='.6f', cbar_kws={'label': 'RMSE'})
    
    # Add title
    plt.title('RMSE Heat Map by Algorithm and Shuffle', fontsize=14)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    print(f"Heat map saved to {output_file}")
    
    # Close the figure
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create RMSE visualization plots.')
    parser.add_argument('--input', default='shuffle_analysis/rmse_results.csv',
                        help='Input CSV file with RMSE values')
    parser.add_argument('--output-dir', default='shuffle_analysis/plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load RMSE data
    df = load_rmse_data(args.input)
    
    # Create standard plots
    box_plot_file = os.path.join(args.output_dir, 'rmse_boxplot.png')
    create_box_plot(df, box_plot_file)
    
    line_plot_file = os.path.join(args.output_dir, 'rmse_lineplot.png')
    create_line_plot(df, line_plot_file)
    
    heat_map_file = os.path.join(args.output_dir, 'rmse_heatmap.png')
    create_heat_map(df, heat_map_file)
    
    # Create additional visualization plots as requested
    # 1. Log-scale line plot
    log_plot_file = os.path.join(args.output_dir, 'rmse_log_lineplot.png')
    create_log_line_plot(df, log_plot_file)
    
    # 2. Broken axis line plot
    broken_axis_file = os.path.join(args.output_dir, 'rmse_broken_axis.png')
    create_broken_axis_plot(df, broken_axis_file)
    
    # 3. Relative performance line plot
    relative_plot_file = os.path.join(args.output_dir, 'rmse_relative.png')
    create_relative_plot(df, relative_plot_file)
    
    print("All plots created successfully!")


if __name__ == '__main__':
    main()
