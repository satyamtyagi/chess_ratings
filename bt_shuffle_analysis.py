#!/usr/bin/env python3
"""
Bradley-Terry Shuffle Robustness Analysis

Modified version that captures individual shuffle results instead of aggregated statistics.
Outputs detailed CSV with RMSE/Tau for each shuffle run with seed tracking.
"""

import argparse
import csv
import random
from typing import List, Tuple, Dict
import numpy as np
from scipy.stats import spearmanr, kendalltau

# Import required functions from main script - need to copy them since import * doesn't work well
import pandas as pd

def load_games_csv(filepath: str) -> List[Tuple[str, str, bool]]:
    """Load games from CSV file."""
    matches = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player1 = str(row['player1'])
            player2 = str(row['player2'])
            winner = str(row['winner'])
            matches.append((player1, player2, winner == player1))
    return matches

def load_true_ratings_csv(filepath: str) -> Dict[str, float]:
    """Load true ratings from CSV file."""
    ratings = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = str(row['player_id'])
            theta = float(row['theta'])
            ratings[player_id] = theta
    return ratings

# Global compute stats tracking (simplified)
_compute_stats = {'sigmoid_evals': 0}

def reset_compute_stats():
    """Reset computation statistics."""
    _compute_stats['sigmoid_evals'] = 0

# Placeholder algorithm functions - will run via subprocess instead
def run_algorithm_subprocess(algorithm_name: str, csv_file: str, true_ratings_file: str, seed: int) -> Tuple[float, float]:
    """Run algorithm via subprocess and extract results."""
    import subprocess
    import re
    import tempfile
    import os
    
    # Create temporary shuffled file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        temp_csv = tmp_file.name
    
    try:
        # Load and shuffle data
        matches = load_games_csv(csv_file)
        rng = random.Random(seed)
        rng.shuffle(matches)
        
        # Write shuffled data
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['player1', 'player2', 'winner'])
            writer.writeheader()
            for p1, p2, p1_won in matches:
                writer.writerow({
                    'player1': p1,
                    'player2': p2, 
                    'winner': p1 if p1_won else p2
                })
        
        # Run algorithm
        cmd = [
            'python', 'bt_single_pass_with_dg.py',
            '--csv', temp_csv,
            '--true-ratings', true_ratings_file,
            '--repeats', '1',  # Single run since we already shuffled
            '--no-table'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return float('nan'), float('nan')
        
        # Extract results from output
        pattern = r'\[([^\]]+)\]\s+RMSE vs Gold: ([0-9\.]+) ± [0-9\.]+;\s+τ: ([0-9\.]+) ± [0-9\.]+;'
        matches = re.findall(pattern, result.stdout)
        
        for alg, rmse, tau in matches:
            if alg.strip() == algorithm_name:
                return float(rmse), float(tau)
        
        return float('nan'), float('nan')
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def evaluate_method_detailed(name: str, matches: List[Tuple[str,str,bool]], players: List[str], 
                           a_gold: Dict[str,float], build_fn, repeats: int = 10, seed: int = 0):
    """Evaluate a rating method and return detailed results for each shuffle."""
    rng = random.Random(seed)
    N = len(matches)
    
    detailed_results = []
    
    for r in range(repeats):
        # Generate shuffle order and track seed
        shuffle_seed = seed + r * 1000  # Ensure different seeds
        shuffle_rng = random.Random(shuffle_seed)
        
        order = list(range(N))
        shuffle_rng.shuffle(order)
        ordered = [matches[i] for i in order]

        # Reset compute counters for this run
        reset_compute_stats()
        
        # Run the algorithm on this shuffled order
        result = build_fn(ordered, players)
        
        # Compute evaluation metrics
        v_gold = np.array([a_gold[p] for p in players])
        v_hat = np.array([result[p] for p in players])
        
        # Center both for RMSE calculation
        v_gold_centered = v_gold - np.mean(v_gold)
        v_hat_centered = v_hat - np.mean(v_hat)
        rmse = np.sqrt(np.mean((v_hat_centered - v_gold_centered)**2))
        
        # Kendall tau and Spearman rho
        try:
            tau = float(kendalltau(v_gold, v_hat)[0])
        except:
            tau = float('nan')
        
        try:
            rho = float(spearmanr(v_gold, v_hat)[0])
        except:
            rho = float('nan')
        
        # Store detailed result
        detailed_results.append({
            'Algorithm': name,
            'Shuffle_Run': r + 1,
            'Seed': shuffle_seed,
            'RMSE': rmse,
            'Kendall_Tau': tau,
            'Spearman_Rho': rho
        })
        
        print(f"  {name} - Shuffle {r+1}/10: RMSE={rmse:.6f}, τ={tau:.6f}, ρ={rho:.6f}")
    
    return detailed_results

def run_shuffle_analysis(csv_file: str, true_ratings_file: str, repeats: int = 10, 
                        base_seed: int = 0, output_file: str = "shuffle_results.csv"):
    """Run shuffle analysis on all algorithms and save detailed results."""
    
    print(f"Loading dataset: {csv_file}")
    print(f"Loading true ratings: {true_ratings_file}")
    print(f"Running {repeats} shuffles per algorithm")
    print()
    
    # Load data
    matches = load_games_csv(csv_file)
    players = sorted(set(p for p1, p2, _ in matches for p in [p1, p2]))
    
    # Load ground truth ratings
    a_gold = load_true_ratings_csv(true_ratings_file)
    print(f"Loaded {len(matches)} games between {len(players)} players")
    print(f"Using true Bradley-Terry thetas as gold standard (mean: {np.mean(list(a_gold.values())):.3f})")
    print()
    
    # Define algorithms to test
    algorithms = [
        ("Batch BT", lambda ms, ps: batch_bt_mle(ms, max_iters=1000, threshold=1e-10)),
        ("ISGD", lambda ms, ps: onepass_isgd(ms, ps, eta=0.1, newton_steps=2, l2=1e-3)),
        ("FTRL-Prox", lambda ms, ps: onepass_ftrl(ms, ps, alpha=0.1, l1=0.0, l2=1e-3)),
        ("Diag-Newton", lambda ms, ps: onepass_diag_newton(ms, ps, ridge=1e-2, step_cap=0.1)),
        ("OPDN-5", lambda ms, ps: build_opdn(ms, ps, sweeps=5)),
        ("DirtyGraph", lambda ms, ps: onepass_dirty_graph(ms, ps, learning_rate=len(ps)/len(ms), phase2_iters=1, edge_cap=None)),
        ("BT-SGD-Counts-15", lambda ms, ps: bt_sgd_counts_wrapper(ms, ps, epochs=15, eta=None)),
        ("BT-SGD-Duels-15", lambda ms, ps: bt_sgd_duels_wrapper(ms, ps, epochs=15, eta=0.005)),
    ]
    
    # Collect all detailed results
    all_results = []
    
    for i, (name, build_fn) in enumerate(algorithms):
        print(f"Evaluating {name}...")
        algorithm_seed = base_seed + i * 10000  # Separate seed ranges for each algorithm
        
        detailed_results = evaluate_method_detailed(
            name=name,
            matches=matches,
            players=players, 
            a_gold=a_gold,
            build_fn=build_fn,
            repeats=repeats,
            seed=algorithm_seed
        )
        
        all_results.extend(detailed_results)
        print()
    
    # Save results to CSV
    print(f"Saving detailed results to {output_file}")
    
    fieldnames = ['Algorithm', 'Shuffle_Run', 'Seed', 'RMSE', 'Kendall_Tau', 'Spearman_Rho']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"✅ Saved {len(all_results)} detailed shuffle results to {output_file}")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print("="*60)
    
    algorithms_list = [name for name, _ in algorithms]
    
    for alg in algorithms_list:
        alg_results = [r for r in all_results if r['Algorithm'] == alg]
        if alg_results:
            rmses = [r['RMSE'] for r in alg_results]
            taus = [r['Kendall_Tau'] for r in alg_results if not np.isnan(r['Kendall_Tau'])]
            
            rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
            tau_mean, tau_std = np.mean(taus), np.std(taus) if taus else (float('nan'), float('nan'))
            
            print(f"{alg:20} RMSE: {rmse_mean:.6f} ± {rmse_std:.6f}, τ: {tau_mean:.6f} ± {tau_std:.6f}")

def main():
    parser = argparse.ArgumentParser(
        description='Bradley-Terry Shuffle Robustness Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bt_shuffle_analysis.py --csv datasets/bt_20k_games.csv --true-ratings datasets/player_bt_20k_thetas.csv
  python bt_shuffle_analysis.py --csv datasets/bt_10k_games.csv --true-ratings datasets/player_bt_20k_thetas.csv --repeats 20
        """
    )
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Input CSV file with BT games')
    parser.add_argument('--true-ratings', type=str, required=True,
                       help='CSV file with true player ratings')
    parser.add_argument('--repeats', type=int, default=10,
                       help='Number of shuffle repeats per algorithm (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--output', type=str, default='shuffle_results.csv',
                       help='Output CSV file for detailed results (default: shuffle_results.csv)')
    
    args = parser.parse_args()
    
    run_shuffle_analysis(
        csv_file=args.csv,
        true_ratings_file=args.true_ratings,
        repeats=args.repeats,
        base_seed=args.seed,
        output_file=args.output
    )

if __name__ == '__main__':
    main()
