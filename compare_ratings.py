#!/usr/bin/env python3
"""
Rating System Comparison

This script compares two rating files by calculating deviation metrics between them.
It reads both CSV files and computes mean square error between the rating values.
"""

import argparse
import csv
import math
import numpy as np


def load_ratings_from_csv(file_path):
    """
    Load ratings from a CSV file.
    
    Args:
        file_path (str): Path to CSV file containing ratings
        
    Returns:
        dict: Dictionary mapping player IDs to their ratings and win counts if available
    """
    ratings = {}
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        
        # Determine column names based on the headers present in the file
        player_col = 'player_no' if 'player_no' in header else 'player'
        elo_col = 'elo_rating' if 'elo_rating' in header else 'elo'
        
        for row in reader:
            player = int(row[player_col])
            bt_rating = float(row['bt_rating'])
            elo_rating = float(row[elo_col])
            
            # Check if wins column exists
            wins = int(float(row['wins'])) if 'wins' in row else 0
            
            ratings[player] = (bt_rating, elo_rating, wins)
    
    return ratings


def compute_mse(ratings1, ratings2, column_index=0):
    """
    Compute Mean Square Error between two sets of ratings.
    
    Args:
        ratings1 (dict): First set of ratings
        ratings2 (dict): Second set of ratings
        column_index (int): Index of the rating column to compare (0=bt_rating, 1=elo_rating)
        
    Returns:
        tuple: (Mean Square Error, Root Mean Square Error)
    """
    players = list(ratings1.keys())
    values1 = [ratings1[p][column_index] for p in players]
    values2 = [ratings2[p][column_index] for p in players]
    
    # Calculate MSE
    mse = sum((v1 - v2) ** 2 for v1, v2 in zip(values1, values2)) / len(players)
    
    # Calculate RMSE
    rmse = math.sqrt(mse)
    
    return mse, rmse


def main():
    parser = argparse.ArgumentParser(description='Compare two rating files and calculate deviations.')
    parser.add_argument('-f1', '--file1', type=str, default='ratings.csv',
                        help='First ratings file (default: ratings.csv)')
    parser.add_argument('-f2', '--file2', type=str, default='elo_ratings.csv',
                        help='Second ratings file (default: elo_ratings.csv)')
    
    args = parser.parse_args()
    
    # Load ratings
    print(f"Loading ratings from {args.file1}...")
    ratings1 = load_ratings_from_csv(args.file1)
    
    print(f"Loading ratings from {args.file2}...")
    ratings2 = load_ratings_from_csv(args.file2)
    
    # Calculate MSE for BT ratings (column index 0)
    bt_mse, bt_rmse = compute_mse(ratings1, ratings2, 0)
    print(f"\nBradley-Terry Rating Comparison:")
    print(f"Mean Square Error: {bt_mse:.6f}")
    print(f"Root Mean Square Error: {bt_rmse:.6f}")
    
    # Calculate MSE for ELO ratings (column index 1)
    elo_mse, elo_rmse = compute_mse(ratings1, ratings2, 1)
    print(f"\nELO Rating Comparison:")
    print(f"Mean Square Error: {elo_mse:.6f}")
    print(f"Root Mean Square Error: {elo_rmse:.6f}")
    
    # Calculate correlations
    players = list(ratings1.keys())
    bt_values1 = [ratings1[p][0] for p in players]  # BT ratings from file 1
    bt_values2 = [ratings2[p][0] for p in players]  # BT ratings from file 2
    elo_values1 = [ratings1[p][1] for p in players]  # ELO ratings from file 1
    elo_values2 = [ratings2[p][1] for p in players]  # ELO ratings from file 2
    
    bt_correlation = np.corrcoef(bt_values1, bt_values2)[0, 1]
    elo_correlation = np.corrcoef(elo_values1, elo_values2)[0, 1]
    
    print(f"\nCorrelations:")
    print(f"Bradley-Terry correlation: {bt_correlation:.6f}")
    print(f"ELO correlation: {elo_correlation:.6f}")
    
    # Print player-by-player comparison
    print("\nPlayer-by-Player Comparison:")
    print("-" * 80)
    print(f"{'Player':6} | {'BT 1':10} | {'BT 2':10} | {'BT Diff':10} | {'ELO 1':10} | {'ELO 2':10} | {'ELO Diff':10} | {'Wins':5}")
    print("-" * 80)
    
    # Sort players by player ID (1 to 10)
    sorted_players = sorted(players)
    
    for player in sorted_players:
        bt1 = ratings1[player][0]
        bt2 = ratings2[player][0]
        bt_diff = bt1 - bt2
        
        elo1 = ratings1[player][1]
        elo2 = ratings2[player][1]
        elo_diff = elo1 - elo2
        
        # Get win count from the first file
        wins = ratings1[player][2]
        
        print(f"{player:6} | {bt1:10.6f} | {bt2:10.6f} | {bt_diff:10.6f} | {elo1:10.1f} | {elo2:10.1f} | {elo_diff:10.1f} | {wins:5d}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average absolute BT difference: {np.mean([abs(ratings1[p][0] - ratings2[p][0]) for p in players]):.6f}")
    print(f"Average absolute ELO difference: {np.mean([abs(ratings1[p][1] - ratings2[p][1]) for p in players]):.1f}")


if __name__ == '__main__':
    main()
