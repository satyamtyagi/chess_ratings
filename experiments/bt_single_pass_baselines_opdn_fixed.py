#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of various single-pass algorithms for Bradley-Terry rating.
This version has improved OPDN implementation with cleaner syntax.
"""

import argparse
import csv
import math
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

def sigmoid(x: float) -> float:
    """Sigmoid function with numerical stability."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def mean_center(a: Dict[str,float]) -> Dict[str,float]:
    """Mean-center a dictionary of values."""
    values = list(a.values())
    m = sum(values) / len(values)
    return {k: v - m for k, v in a.items()}


def read_matches(filename: str) -> List[Tuple[str,str,bool]]:
    """Read matches from CSV file."""
    matches = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'player_first' in row and 'player_second' in row:  # New column names
                p1, p2 = row['player_first'], row['player_second']
                # Handle potential string values like 'w' for win
                if 'result' in row:
                    w1 = 1 if row['result'] in ['1', 'w', 'win'] else 0
                elif 'win' in row:
                    w1 = 1 if row['win'] in ['1', 'w', 'win'] else 0
                else:
                    w1 = 1  # Default to win for player 1
            else:  # Old column names
                p1, p2 = row['player_i'], row['player_j']
                # Handle potential string values
                if 'result' in row:
                    w1 = 1 if row['result'] in ['1', 'w', 'win'] else 0
                elif 'win' in row:
                    w1 = 1 if row['win'] in ['1', 'w', 'win'] else 0
                else:
                    w1 = 1  # Default to win for player 1
            matches.append((p1, p2, w1 == 1))
    return matches


def save_ratings_to_csv(ratings: Dict[str,float], wins_count: Dict[str,int], filename: str) -> None:
    """Save ratings to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['player_no', 'bt_rating', 'elo_rating', 'wins'])
        
        # Convert to ELO scale
        factor = 400.0 / math.log(10.0)
        elo_ratings = {p: 1200.0 + factor * r for p, r in ratings.items()}
        
        # Sort by player number
        sorted_players = sorted(ratings.keys(), key=lambda p: int(p) if p.isdigit() else p)
        
        for p in sorted_players:
            writer.writerow([p, f"{ratings[p]:.6f}", f"{elo_ratings[p]:.1f}", wins_count[p]])
    print(f"Ratings saved to {filename}")


# ===== Batch BT via multiplicative weights (Minorization-Maximization, not gradient descent) =====

def batch_bt_mle(matches: List[Tuple[str,str,bool]], players: List[str]) -> Dict[str,float]:
    """Batch Bradley-Terry MLE via multiplicative weights."""
    # Count wins
    wins = {p: defaultdict(int) for p in players}
    for p1, p2, w1 in matches:
        if w1:
            wins[p1][p2] += 1
        else:
            wins[p2][p1] += 1

    # Initialize ratings in real space (not log space)
    w = {p: 1.0 for p in players}
    eps = 1e-12
    for it in range(1000):
        max_rel = 0.0
        w_new = {}
        for i in players:
            num = sum(wins[i].values()) + eps
            denom = 0.0
            for j in players:
                if j == i: continue
                m_ij = wins[i].get(j,0) + wins[j].get(i,0)
                if m_ij == 0: continue
                denom += m_ij * (w[i] / (w[i] + w[j]))
            denom += eps
            wi_new = w[i] * (num / denom)
            w_new[i] = max(wi_new, 1e-15)
            max_rel = max(max_rel, abs(w_new[i]-w[i])/(w[i]+1e-15))
        w = w_new
        print(f"Iteration {it+1}: max_rel = {max_rel}")
        if max_rel < 1e-10:
            print("Converged!")
            break

    # Convert from real space to log space
    a = {p: math.log(w[p]) for p in players}
    
    # Mean-center
    return mean_center(a)


# ----------------- Single-pass optimizers -----------------

def onepass_isgd(matches: List[Tuple[str,str,bool]], players: List[str], eta: float = 0.1, newton_steps: int = 2, l2: float = 1e-3) -> Dict[str,float]:
    """Implicit SGD for BT; one pass in given order; tiny L2; recenter at end."""
    a = {p: 0.0 for p in players}
    for p1, p2, w1 in matches:
        y = 1.0 if w1 else 0.0
        d = a[p1] - a[p2]

        # Solve Δ' = d + 2η( y - σ(Δ') ) via up to 'newton_steps' Newton iterations
        dp = d
        for _ in range(newton_steps):
            p = sigmoid(dp)
            g = dp - d - 2*eta*(y - p)         # f(dp) = dp - d - 2η(y - σ(dp)) = 0
            h = 1.0 - 2*eta*p*(1-p)            # f'(dp) = 1 - 2η σ'(dp)
            dp = dp - g / max(h, 1e-8)
        p = sigmoid(dp)
        grad = (y - p)

        # Update (including L2 regularization, although p1 and p2 will have same regularization)
        update = eta * grad
        a[p1] = a[p1]*(1.0 - eta*l2) + update
        a[p2] = a[p2]*(1.0 - eta*l2) - update
    
    return mean_center(a)


def onepass_ftrl(matches: List[Tuple[str,str,bool]], players: List[str], alpha: float = 0.1, l1: float = 0.0, l2: float = 1e-3) -> Dict[str,float]:
    """Per-Coordinate FTRL-Proximal for BT; one pass in given order; L1+L2; recenter at end."""
    z = {p: 0.0 for p in players}
    n = {p: 0.0 for p in players}
    a = {p: 0.0 for p in players}
    
    # FTRL-Proximal with per-coordinate learning rates
    for p1, p2, w1 in matches:
        y = 1.0 if w1 else 0.0
        d = a[p1] - a[p2]
        p = sigmoid(d)
        g = p - y
        
        # Per-coordinate updates for p1
        sigma1 = 1.0 / alpha * (math.sqrt(n[p1] + g*g) - math.sqrt(n[p1]))
        z[p1] = z[p1] + g - sigma1 * a[p1]
        n[p1] = n[p1] + g*g
        
        # Per-coordinate updates for p2
        sigma2 = 1.0 / alpha * (math.sqrt(n[p2] + g*g) - math.sqrt(n[p2]))
        z[p2] = z[p2] - g - sigma2 * a[p2]
        n[p2] = n[p2] + g*g
        
        # Compute new weights
        for p in [p1, p2]:
            if abs(z[p]) <= l1:
                a[p] = 0.0
            else:
                a[p] = (-z[p] + np.sign(z[p])*l1) / (l2 + 1.0 / alpha * math.sqrt(n[p]))
    
    return mean_center(a)


def onepass_diag_newton(matches: List[Tuple[str,str,bool]], players: List[str], ridge: float = 1e-2, step_cap: Optional[float] = 0.1) -> Dict[str,float]:
    """Diagonal Newton (H only has diagonal entries) for BT; one pass in given order; recenter at end."""
    a = {p: 0.0 for p in players}
    h = {p: ridge for p in players}  # start with small ridge
    
    for p1, p2, w1 in matches:
        y = 1.0 if w1 else 0.0
        d = a[p1] - a[p2]
        p = sigmoid(d)
        g1 = p - y
        g2 = p * (1.0 - p)  # d sigmoid(d) / d d
        
        # Update Hessian
        h[p1] = h[p1] + g2
        h[p2] = h[p2] + g2
        
        # Update parameters
        step1 = g1 / h[p1]
        step2 = g1 / h[p2]
        
        # Optional step capping
        if step_cap is not None:
            step1 = min(max(-step_cap, step1), step_cap)
            step2 = min(max(-step_cap, step2), step_cap)
            
        a[p1] = a[p1] - step1
        a[p2] = a[p2] + step2
    
    return mean_center(a)


def one_pass_opdn_counts(pair_counts, n_players, ridge=1e-2, step_cap=None, recenter=True):
    """One-Pass Diagonal Newton for BT with symmetric updates and pair counts."""
    a = np.zeros(n_players)
    
    # Process each pair
    for (i, j), (n_ij, w_i) in pair_counts.items():
        # Skip pairs with no matches
        if n_ij == 0:
            continue
            
        # Calculate prediction and step
        d_ij = a[i] - a[j]
        p_ij = sigmoid(d_ij)
        
        # Calculate gradients and Hessians
        g_ij = p_ij - w_i / n_ij
        h_ij = p_ij * (1 - p_ij)
        
        # Compute step size with symmetric regularization
        denom = 2 * n_ij * h_ij + 2 * ridge
        if denom < 1e-10:  # Safety check
            denom = 1e-10
            
        step = n_ij * g_ij / denom
        
        # Optional step capping
        if step_cap is not None:
            step = min(max(-step_cap, step), step_cap)
        
        # Make symmetric updates
        a[i] -= step
        a[j] += step
    
    # Recenter
    if recenter:
        a -= np.mean(a)
        
    return a


def onepass_opdn(matches, players, ridge=1e-2, step_cap=None):
    """Clean implementation of One-Pass Diagonal Newton for BT with symmetric updates."""
    # Create player index mapping
    player_to_idx = {p: i for i, p in enumerate(players)}
    
    # Count pairs
    pairs = {}
    for p1, p2, w1 in matches:
        i = min(player_to_idx[p1], player_to_idx[p2])
        j = max(player_to_idx[p1], player_to_idx[p2])
        
        # Update pair counts
        if (i, j) not in pairs:
            pairs[(i, j)] = [0, 0]  # [n_ij, w_i]
            
        pairs[(i, j)][0] += 1  # increment match count
        
        # Increment win count for player i
        if (player_to_idx[p1] < player_to_idx[p2] and w1) or (player_to_idx[p1] > player_to_idx[p2] and not w1):
            pairs[(i, j)][1] += 1
    
    # Run OPDN algorithm
    ratings = one_pass_opdn_counts(pairs, len(players), ridge, step_cap, recenter=True)
    
    # Return as player dictionary
    return {p: r for p, r in zip(players, ratings)}


def main():
    ap = argparse.ArgumentParser(description="Single-pass BT baselines vs batch BT (gold).")
    ap.add_argument("--csv", required=True, help="Path to CSV (game, p1, p2, W/L).")
    ap.add_argument("--repeats", type=int, default=10, help="Number of random orders per method.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for shuffles.")
    ap.add_argument("--no-table", action="store_true", help="Don't show per-player tables for each method.")
    ap.add_argument("--save-csv", action="store_true", help="Save ratings to CSV files.")
    ap.add_argument("--out-prefix", type=str, default="singlepass_", help="Prefix for output CSV files.")
    args = ap.parse_args()

    # Always use all matches
    matches = read_matches(args.csv)

    players = sorted({p for m in matches for p in (m[0], m[1])}, key=lambda x: int(x) if x.isdigit() else x)

    # Run batch_bt_mle to get "gold" ratings
    print("Computing batch BT (gold) ratings...")
    a_gold = batch_bt_mle(matches, players)
    
    # Define algorithms to test
    algorithm_builders = [
        ("ISGD", lambda: onepass_isgd(matches, players, eta=0.1, newton_steps=2, l2=1e-3)),
        ("FTRL-Prox", lambda: onepass_ftrl(matches, players, alpha=0.1, l1=0.0, l2=1e-3)),
        ("Diag-Newton", lambda: onepass_diag_newton(matches, players, ridge=1e-2, step_cap=0.1)),
        ("OPDN", lambda: onepass_opdn(matches, players, ridge=0.0, step_cap=None)),
    ]
    
    # Run algorithms and process results
    
    # Calculate win counts for batch_bt_mle (used for both display and CSV)
    wins_count_bt = defaultdict(int)
    for p1, p2, w1 in matches:
        if w1: wins_count_bt[p1] += 1
        else:  wins_count_bt[p2] += 1
    
    # Calculate ELO ratings for batch_bt_mle
    factor = 400.0 / math.log(10.0)
    elo_bt = {p: 1200.0 + factor * a_gold[p] for p in players}
    
    # Display batch_bt_mle table if requested
    if not args.no_table:
        print("\n-------------------------------------------")
        print(f"Batch BT (Gold Standard) — Full table (subset size = {len(matches)})")
        print("-------------------------------------------")
        print("Rank | Player |   θ (centered)  |   Elo   | Wins")
        print("-------------------------------------------")
        
        # Display batch_bt_mle table, sorted by player number but showing true rank
        order_bt = sorted(players, key=lambda p: int(p) if p.isdigit() else p)
        # Calculate ranks
        ranks = {p: idx+1 for idx, p in enumerate(sorted(players, key=lambda p: a_gold[p], reverse=True))}
        for p in order_bt:
            print(f"{ranks[p]:4d} | {int(p):6d} | {a_gold[p]:13.6f} | {elo_bt[p]:7.1f} | {wins_count_bt[p]:4d}")
    
    # Save batch_bt_mle to CSV if requested
    if args.save_csv:
        output_file = f"{args.out_prefix}batch_bt_ratings.csv"
        save_ratings_to_csv(a_gold, wins_count_bt, output_file)
    
    # Run each algorithm, generate results, and optionally show tables or save CSVs
    for name, builder in algorithm_builders:
        # This is where the actual algorithm runs - must run regardless of display options
        a_hat = builder()
        
        # Calculate win counts
        wins_count = defaultdict(int)
        for p1, p2, w1 in matches:
            if w1: wins_count[p1] += 1
            else:  wins_count[p2] += 1
        
        # Calculate ELO ratings
        factor = 400.0 / math.log(10.0)
        elo = {p: 1200.0 + factor * a_hat[p] for p in players}
        
        # Save to CSV if requested
        if args.save_csv:
            output_file = f"{args.out_prefix}{name.lower().replace('-', '_')}_ratings.csv"
            save_ratings_to_csv(a_hat, wins_count, output_file)
        
        # Display table unless explicitly disabled
        if not args.no_table:
            order = sorted(players, key=lambda p: int(p) if p.isdigit() else p)
            print("\n-------------------------------------------")
            print(f"{name} — One-pass table (subset size = {len(matches)}), sorted by player number)")
            print("-------------------------------------------")
            print("Rank | Player |   θ (centered)  |   Elo   | Wins")
            print("-------------------------------------------")
            # Calculate ranks
            ranks = {p: idx+1 for idx, p in enumerate(sorted(players, key=lambda p: a_hat[p], reverse=True))}
            for p in order:
                print(f"{ranks[p]:4d} | {int(p):6d} | {a_hat[p]:13.6f} | {elo[p]:7.1f} | {wins_count[p]:4d}")


if __name__ == "__main__":
    main()
