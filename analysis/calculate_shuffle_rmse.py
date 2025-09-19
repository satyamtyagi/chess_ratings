#!/usr/bin/env python3
"""
Shuffle Analysis RMSE Calculator

This script calculates RMSE values between Bradley-Terry and other rating algorithms
across multiple shuffled datasets.
"""

import argparse
import csv
import math
import os
import glob
import numpy as np
import pandas as pd


def load_ratings_from_csv(file_path):
    """
    Load ratings from a CSV file.
    
    Args:
        file_path (str): Path to CSV file containing ratings
        
    Returns:
        dict: Dictionary mapping player IDs to their ratings
    """
    ratings = {}
    
    # Based on our examination of the files, we know the exact column names used:
    # Bradley-Terry: player_no, bt_rating
    # Dirty Graph: player, bt_rating
    # Single-Pass: player_no, bt_rating
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Get header row
        
        # Identify the correct column indices
        player_col_idx = -1
        rating_col_idx = -1
        
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if 'player_no' == header_lower or 'player' == header_lower:
                player_col_idx = i
            elif 'bt_rating' == header_lower or 'bt (θ)' == header_lower:
                rating_col_idx = i
        
        if player_col_idx == -1 or rating_col_idx == -1:
            raise ValueError(f"Could not find player or rating columns in {file_path}")
        
        # Read rows and extract player IDs and ratings
        for row in reader:
            if not row:  # Skip empty rows
                continue
            
            try:
                player = int(float(row[player_col_idx]))
                rating = float(row[rating_col_idx])
                ratings[player] = rating
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse row {row}: {e}")
    
    return ratings


def compute_rmse(ratings1, ratings2):
    """
    Compute Root Mean Square Error between two sets of ratings.
    
    Args:
        ratings1 (dict): First set of ratings
        ratings2 (dict): Second set of ratings
        
    Returns:
        float: Root Mean Square Error
    """
    players = list(ratings1.keys())
    values1 = [ratings1[p] for p in players]
    values2 = [ratings2[p] for p in players]
    
    # Calculate MSE
    mse = sum((v1 - v2) ** 2 for v1, v2 in zip(values1, values2)) / len(players)
    
    # Calculate RMSE
    rmse = math.sqrt(mse)
    
    return rmse


def process_shuffle(shuffle_num, bt_dir, dg_dir, sp_dir):
    """
    Process a single shuffle and calculate RMSE values.
    
    Args:
        shuffle_num (str): Shuffle number (e.g., '01')
        bt_dir (str): Directory containing Bradley-Terry results
        dg_dir (str): Directory containing Dirty Graph results
        sp_dir (str): Directory containing Single-Pass results
        
    Returns:
        dict: Dictionary of RMSE values
    """
    # Load Bradley-Terry ratings
    bt_file = os.path.join(bt_dir, f"bt_shuffle_{shuffle_num}.csv")
    bt_ratings = load_ratings_from_csv(bt_file)
    
    # Load Dirty Graph ratings
    dg_file = os.path.join(dg_dir, f"dg_shuffle_{shuffle_num}.csv")
    dg_ratings = load_ratings_from_csv(dg_file)
    
    # Load Single-Pass ratings
    isgd_file = os.path.join(sp_dir, f"sp_shuffle_{shuffle_num}_isgd_ratings.csv")
    ftrl_file = os.path.join(sp_dir, f"sp_shuffle_{shuffle_num}_ftrl_prox_ratings.csv")
    diag_file = os.path.join(sp_dir, f"sp_shuffle_{shuffle_num}_diag_newton_ratings.csv")
    
    isgd_ratings = load_ratings_from_csv(isgd_file)
    ftrl_ratings = load_ratings_from_csv(ftrl_file)
    diag_ratings = load_ratings_from_csv(diag_file)
    
    # Calculate RMSE values
    dg_rmse = compute_rmse(bt_ratings, dg_ratings)
    isgd_rmse = compute_rmse(bt_ratings, isgd_ratings)
    ftrl_rmse = compute_rmse(bt_ratings, ftrl_ratings)
    diag_rmse = compute_rmse(bt_ratings, diag_ratings)
    
    return {
        'shuffle': int(shuffle_num),
        'DG': dg_rmse,
        'ISGD': isgd_rmse,
        'FTRL': ftrl_rmse,
        'DiagNewton': diag_rmse
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate RMSE values across shuffled datasets.')
    parser.add_argument('--bt-dir', default='shuffle_analysis/bt_results',
                        help='Directory containing Bradley-Terry results')
    parser.add_argument('--dg-dir', default='shuffle_analysis/dg_results',
                        help='Directory containing Dirty Graph results')
    parser.add_argument('--sp-dir', default='shuffle_analysis/singlepass_results',
                        help='Directory containing Single-Pass results')
    parser.add_argument('--out-file', default='shuffle_analysis/rmse_results.csv',
                        help='Output CSV file for RMSE values')
    parser.add_argument('--num-shuffles', type=int, default=10,
                        help='Number of shuffled datasets')
    args = parser.parse_args()
    
    # Process all shuffles
    results = []
    for i in range(1, args.num_shuffles + 1):
        shuffle_num = f"{i:02d}"
        try:
            rmse_values = process_shuffle(shuffle_num, args.bt_dir, args.dg_dir, args.sp_dir)
            results.append(rmse_values)
            print(f"Processed shuffle {shuffle_num}")
        except Exception as e:
            print(f"Error processing shuffle {shuffle_num}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate statistics
    means = df.mean()
    stds = df.std()
    mins = df.min()
    maxs = df.max()
    
    # Print summary statistics
    print("\nRMSE Statistics vs. Bradley-Terry:")
    print("-----------------------------------")
    print(f"Algorithm   | Mean ± Std      | Min      | Max")
    print("-----------------------------------")
    for algo in ['DG', 'ISGD', 'FTRL', 'DiagNewton']:
        print(f"{algo:11} | {means[algo]:.6f} ± {stds[algo]:.6f} | {mins[algo]:.6f} | {maxs[algo]:.6f}")
    
    # Save to CSV
    df.to_csv(args.out_file, index=False)
    print(f"\nRMSE values saved to {args.out_file}")


if __name__ == '__main__':
    main()
