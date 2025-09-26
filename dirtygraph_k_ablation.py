#!/usr/bin/env python3
"""
DirtyGraph K-value ablation study.
Test different edge-cap (K) values and capture RMSE and Kendall Tau results.
"""

import subprocess
import re
import csv
import argparse
from typing import Dict, Optional

def run_single_k_analysis(csv_file: str, true_ratings_file: str, k_value: int, seed: int = 42) -> Optional[Dict]:
    """Run analysis with a specific K value and extract DirtyGraph results."""
    
    # Handle "infinite" case with very large K
    k_str = str(k_value) if k_value != -1 else "999999"
    k_label = str(k_value) if k_value != -1 else "infinite"
    
    cmd = [
        'python', 'bt_single_pass_with_dg.py',
        '--csv', csv_file
    ]
    
    # Add true-ratings only if provided
    if true_ratings_file:
        cmd.extend(['--true-ratings', true_ratings_file])
    
    cmd.extend([
        '--repeats', '1',
        '--seed', str(seed),
        '--edge-cap', k_str,
        '--no-table'
    ])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"Error with K={k_label}: {result.stderr}")
            return None
        
        # Extract DirtyGraph results only (handle nan values)
        pattern = r'\[DirtyGraph\]\s+RMSE vs Gold: ([0-9\.]+) ± [0-9\.]+;\s+τ: (nan|[0-9\.]+) ± (nan|[0-9\.]+);'
        match = re.search(pattern, result.stdout)
        
        if match:
            rmse, tau, _ = match.groups()  # Third group is tau_std, not needed
            tau_value = float('nan') if tau == 'nan' else float(tau)
            return {
                'K': k_label,
                'RMSE': float(rmse),
                'Kendall_Tau': tau_value
            }
        else:
            print(f"Could not parse DirtyGraph results for K={k_label}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Timeout for K={k_label}")
        return None
    except Exception as e:
        print(f"Error running K={k_label}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="DirtyGraph K-value ablation study")
    parser.add_argument("--csv", required=True, help="Path to CSV file with games")
    parser.add_argument("--true-ratings", required=False, help="Path to true ratings CSV (optional, uses Batch BT as gold standard if not provided)")
    parser.add_argument("--output", default="dirtygraph_k_ablation_results.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # K values to test (use -1 to represent infinite)
    k_values = [4, 6, 8, 10, 12, 14, 16, 18, -1]
    
    print("Running DirtyGraph K-value ablation study:")
    print(f"  Dataset: {args.csv}")
    print(f"  True ratings: {args.true_ratings}")
    print(f"  K values: {[str(k) if k != -1 else 'infinite' for k in k_values]}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    print()
    
    results = []
    
    for i, k in enumerate(k_values, 1):
        k_label = str(k) if k != -1 else "infinite"
        print(f"Running {i}/{len(k_values)}: K={k_label}...")
        
        result = run_single_k_analysis(args.csv, args.true_ratings, k, args.seed)
        if result:
            results.append(result)
            print(f"  ✅ K={k_label}: RMSE={result['RMSE']:.6f}, τ={result['Kendall_Tau']:.6f}")
        else:
            print(f"  ❌ K={k_label}: Failed")
        print()
    
    # Save results to CSV
    if results:
        with open(args.output, 'w', newline='') as f:
            fieldnames = ['K', 'RMSE', 'Kendall_Tau']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✅ Saved {len(results)} results to {args.output}")
        
        # Print summary
        print("\n📊 K-Value Ablation Summary:")
        print("=" * 50)
        for result in results:
            print(f"K={result['K']:>8}: RMSE={result['RMSE']:.6f}, τ={result['Kendall_Tau']:.6f}")
        
        # Find best K by RMSE
        best_rmse = min(results, key=lambda x: x['RMSE'])
        best_tau = max(results, key=lambda x: x['Kendall_Tau'])
        print(f"\n🏆 Best RMSE: K={best_rmse['K']} (RMSE={best_rmse['RMSE']:.6f})")
        print(f"🏆 Best Tau:  K={best_tau['K']} (τ={best_tau['Kendall_Tau']:.6f})")
        
    else:
        print("❌ No results collected")

if __name__ == '__main__':
    main()
