#!/usr/bin/env python3
"""
Run incremental analysis on BT dataset subsets and compile results.
"""

import subprocess
import re
import pandas as pd
import os
from pathlib import Path

# Dataset files to analyze
DATASETS = [
    "bt_20k_2k_games.csv",
    "bt_20k_4k_games.csv", 
    "bt_20k_6k_games.csv",
    "bt_20k_8k_games.csv",
    "bt_20k_10k_games.csv",
    "bt_20k_12k_games.csv",
    "bt_20k_14k_games.csv",
    "bt_20k_16k_games.csv",
    "bt_20k_18k_games.csv",
    "bt_20k_20k_games.csv"
]

# Ground truth file
GROUND_TRUTH = "player_bt_20k_thetas.csv"

def extract_results_from_output(output_text):
    """Extract algorithm results from the command output."""
    results = {}
    
    # Pattern to match algorithm results
    pattern = r'\[([^\]]+)\]\s+RMSE vs Gold: ([0-9\.]+) ± [0-9\.]+;\s+τ: ([0-9\.]+) ± [0-9\.]+;'
    
    matches = re.findall(pattern, output_text)
    
    for algorithm, rmse, tau in matches:
        results[algorithm] = {
            'RMSE': float(rmse),
            'Kendall_Tau': float(tau)
        }
    
    return results

def run_analysis_on_dataset(dataset_file):
    """Run the BT analysis on a single dataset file."""
    print(f"Analyzing {dataset_file}...")
    
    cmd = [
        "python", "bt_single_pass_with_dg.py",
        "--csv", f"datasets/{dataset_file}",
        "--opdn-passes", "5",
        "--no-table",
        "--sgd-epochs", "15",
        "--true-ratings", f"datasets/{GROUND_TRUTH}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Error running {dataset_file}: {result.stderr}")
            return None
        
        return extract_results_from_output(result.stdout)
    
    except subprocess.TimeoutExpired:
        print(f"Timeout running {dataset_file}")
        return None
    except Exception as e:
        print(f"Exception running {dataset_file}: {e}")
        return None

def compile_results():
    """Run analysis on all datasets and compile results."""
    all_results = {}
    
    for dataset in DATASETS:
        # Extract dataset size from filename
        size_match = re.search(r'(\d+)k_games\.csv', dataset)
        if size_match:
            size = f"{size_match.group(1)}k"
        else:
            size = dataset
        
        results = run_analysis_on_dataset(dataset)
        if results:
            all_results[size] = results
    
    return all_results

def create_rmse_table(results):
    """Create RMSE comparison table."""
    # Get all algorithms
    algorithms = set()
    for size_results in results.values():
        algorithms.update(size_results.keys())
    
    algorithms = sorted(list(algorithms))
    
    # Create DataFrame
    rmse_data = []
    for size, size_results in results.items():
        row = {'Dataset Size': size}
        for alg in algorithms:
            if alg in size_results:
                row[alg] = f"{size_results[alg]['RMSE']:.6f}"
            else:
                row[alg] = "N/A"
        rmse_data.append(row)
    
    return pd.DataFrame(rmse_data)

def create_tau_table(results):
    """Create Kendall Tau comparison table."""
    # Get all algorithms
    algorithms = set()
    for size_results in results.values():
        algorithms.update(size_results.keys())
    
    algorithms = sorted(list(algorithms))
    
    # Create DataFrame
    tau_data = []
    for size, size_results in results.items():
        row = {'Dataset Size': size}
        for alg in algorithms:
            if alg in size_results:
                row[alg] = f"{size_results[alg]['Kendall_Tau']:.6f}"
            else:
                row[alg] = "N/A"
        tau_data.append(row)
    
    return pd.DataFrame(tau_data)

def main():
    """Main execution function."""
    print("Starting incremental analysis...")
    print("This will run bt_single_pass_with_dg.py on all subset datasets.")
    print(f"Using ground truth: {GROUND_TRUTH}")
    print()
    
    # Compile results
    results = compile_results()
    
    if not results:
        print("No results collected!")
        return
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - GENERATING TABLES")
    print("="*80)
    
    # Create tables
    rmse_table = create_rmse_table(results)
    tau_table = create_tau_table(results)
    
    # Display tables
    print("\n📊 RMSE vs True Player Thetas")
    print("="*50)
    print(rmse_table.to_string(index=False))
    
    print("\n🎯 Kendall Tau (τ) Ranking Agreement")
    print("="*50)
    print(tau_table.to_string(index=False))
    
    # Save to CSV files
    rmse_table.to_csv("incremental_rmse_results.csv", index=False)
    tau_table.to_csv("incremental_tau_results.csv", index=False)
    
    print(f"\n✅ Results saved to:")
    print(f"   - incremental_rmse_results.csv")
    print(f"   - incremental_tau_results.csv")

if __name__ == "__main__":
    main()
