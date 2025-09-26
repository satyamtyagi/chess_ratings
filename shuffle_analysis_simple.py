#!/usr/bin/env python3
"""
Simple Shuffle Analysis - Individual Results

Runs bt_single_pass_with_dg.py multiple times with different seeds
to collect individual shuffle results instead of aggregated statistics.
"""

import subprocess
import re
import csv
import argparse

def run_single_analysis(csv_file: str, true_ratings_file: str, seed: int) -> dict:
    """Run analysis with a specific seed and extract results."""
    cmd = [
        'python', 'bt_single_pass_with_dg.py',
        '--csv', csv_file,
        '--true-ratings', true_ratings_file,
        '--repeats', '1',  # Single run per call
        '--seed', str(seed),
        '--no-table',
        '--opdn-passes', '5',  # CRITICAL: Use 5 passes for OPDN
        '--sgd-epochs', '15'   # CRITICAL: Use 15 epochs for SGD
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"Error with seed {seed}: {result.stderr}")
            return {}
        
        # Extract algorithm results
        pattern = r'\[([^\]]+)\]\s+RMSE vs Gold: ([0-9\.]+) ± [0-9\.]+;\s+τ: ([0-9\.]+) ± [0-9\.]+;'
        matches = re.findall(pattern, result.stdout)
        
        results = {}
        for algorithm, rmse, tau in matches:
            results[algorithm.strip()] = {
                'RMSE': float(rmse),
                'Kendall_Tau': float(tau),
                'Seed': seed
            }
        
        return results
    
    except subprocess.TimeoutExpired:
        print(f"Timeout with seed {seed}")
        return {}
    except Exception as e:
        print(f"Exception with seed {seed}: {e}")
        return {}

def run_shuffle_analysis(csv_file: str, true_ratings_file: str, num_shuffles: int = 10, 
                        base_seed: int = 42, output_file: str = "shuffle_detailed_results.csv"):
    """Run multiple shuffle analyses and collect detailed results."""
    
    print(f"Running shuffle analysis:")
    print(f"  Dataset: {csv_file}")
    print(f"  True ratings: {true_ratings_file}")
    print(f"  Number of shuffles: {num_shuffles}")
    print(f"  Base seed: {base_seed}")
    print(f"  Output file: {output_file}")
    print()
    
    all_results = []
    
    for shuffle_run in range(1, num_shuffles + 1):
        seed = base_seed + shuffle_run * 1000  # Ensure different seeds
        print(f"Running shuffle {shuffle_run}/{num_shuffles} (seed: {seed})...")
        
        run_results = run_single_analysis(csv_file, true_ratings_file, seed)
        
        if run_results:
            for algorithm, metrics in run_results.items():
                all_results.append({
                    'Algorithm': algorithm,
                    'Shuffle_Run': shuffle_run,
                    'Seed': metrics['Seed'],
                    'RMSE': metrics['RMSE'],
                    'Kendall_Tau': metrics['Kendall_Tau']
                })
            print(f"  ✅ Collected results for {len(run_results)} algorithms")
        else:
            print(f"  ❌ No results for shuffle {shuffle_run}")
        print()
    
    # Save detailed results
    if all_results:
        fieldnames = ['Algorithm', 'Shuffle_Run', 'Seed', 'RMSE', 'Kendall_Tau']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"✅ Saved {len(all_results)} detailed results to {output_file}")
        
        # Print summary
        print("\n📊 Summary by Algorithm:")
        print("="*60)
        
        algorithms = list(set(r['Algorithm'] for r in all_results))
        for alg in sorted(algorithms):
            alg_results = [r for r in all_results if r['Algorithm'] == alg]
            rmses = [r['RMSE'] for r in alg_results]
            taus = [r['Kendall_Tau'] for r in alg_results]
            
            rmse_mean = sum(rmses) / len(rmses)
            rmse_std = (sum((x - rmse_mean)**2 for x in rmses) / len(rmses))**0.5
            tau_mean = sum(taus) / len(taus)
            tau_std = (sum((x - tau_mean)**2 for x in taus) / len(taus))**0.5
            
            print(f"{alg:20} RMSE: {rmse_mean:.6f} ± {rmse_std:.6f}, τ: {tau_mean:.6f} ± {tau_std:.6f}")
    else:
        print("❌ No results collected!")

def main():
    parser = argparse.ArgumentParser(
        description='Bradley-Terry Shuffle Analysis - Individual Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shuffle_analysis_simple.py --csv datasets/bt_20k_games.csv --true-ratings datasets/player_bt_20k_thetas.csv
  python shuffle_analysis_simple.py --csv datasets/bt_10k_games.csv --true-ratings datasets/player_bt_20k_thetas.csv --shuffles 20
        """
    )
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Input CSV file with BT games')
    parser.add_argument('--true-ratings', type=str, required=True,
                       help='CSV file with true player ratings')
    parser.add_argument('--shuffles', type=int, default=10,
                       help='Number of shuffle runs (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--output', type=str, default='shuffle_detailed_results.csv',
                       help='Output CSV file (default: shuffle_detailed_results.csv)')
    
    args = parser.parse_args()
    
    run_shuffle_analysis(
        csv_file=args.csv,
        true_ratings_file=args.true_ratings,
        num_shuffles=args.shuffles,
        base_seed=args.seed,
        output_file=args.output
    )

if __name__ == '__main__':
    main()
