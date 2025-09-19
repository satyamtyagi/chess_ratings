#!/usr/bin/env python3
"""
Combine Ratings Script

This script combines the Bradley-Terry and Dirty Graph ratings into a single CSV file
with columns for player ID, BT rating, BT ELO, DG rating, and DG ELO.
"""

import csv
import argparse


def load_ratings_from_csv(file_path):
    """
    Load ratings from a CSV file.
    
    Args:
        file_path (str): Path to CSV file containing ratings
        
    Returns:
        dict: Dictionary mapping player IDs to their ratings
    """
    ratings = {}
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        
        # Determine column names based on file format
        player_col = 'player_no' if 'player_no' in header else 'player'
        bt_col = 'bt_rating' if 'bt_rating' in header else 'bt_rating'
        elo_col = 'elo_rating' if 'elo_rating' in header else 'elo'
        
        for row in reader:
            player_id = int(row[player_col])
            bt_rating = float(row[bt_col])
            elo_rating = float(row[elo_col])
            wins = float(row['wins']) if 'wins' in row else 0.0
            
            ratings[player_id] = {
                'bt_rating': bt_rating,
                'elo_rating': elo_rating,
                'wins': wins
            }
    
    return ratings


def combine_ratings(bt_ratings, dg_ratings, output_file):
    """
    Combine two rating systems into a single CSV file
    
    Args:
        bt_ratings (dict): Bradley-Terry ratings dictionary
        dg_ratings (dict): Dirty Graph ratings dictionary
        output_file (str): Output file path
    """
    # Find all unique player IDs across both systems
    all_players = sorted(set(bt_ratings.keys()).union(set(dg_ratings.keys())))
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['player_no', 'bt_rating', 'bt_elo', 'dg_rating', 'dg_elo', 'wins']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for player in all_players:
            bt_data = bt_ratings.get(player, {'bt_rating': 0.0, 'elo_rating': 1200.0, 'wins': 0.0})
            dg_data = dg_ratings.get(player, {'bt_rating': 0.0, 'elo_rating': 1200.0, 'wins': 0.0})
            
            # Use wins from BT ratings if available, otherwise from DG ratings
            wins = bt_data['wins'] if bt_data['wins'] > 0 else dg_data['wins']
            
            writer.writerow({
                'player_no': player,
                'bt_rating': bt_data['bt_rating'],
                'bt_elo': bt_data['elo_rating'],
                'dg_rating': dg_data['bt_rating'],
                'dg_elo': dg_data['elo_rating'],
                'wins': wins
            })
    
    print(f"Combined ratings written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Combine two rating files into one CSV')
    parser.add_argument('-bt', '--bt_file', default='huge_bt_ratings.csv', 
                        help='Bradley-Terry ratings file (default: huge_bt_ratings.csv)')
    parser.add_argument('-dg', '--dg_file', default='huge_dg_ratings.csv',
                        help='Dirty Graph ratings file (default: huge_dg_ratings.csv)')
    parser.add_argument('-o', '--output', default='combined_ratings.csv',
                        help='Output file name (default: combined_ratings.csv)')
    
    args = parser.parse_args()
    
    print(f"Loading Bradley-Terry ratings from {args.bt_file}")
    bt_ratings = load_ratings_from_csv(args.bt_file)
    
    print(f"Loading Dirty Graph ratings from {args.dg_file}")
    dg_ratings = load_ratings_from_csv(args.dg_file)
    
    print(f"Combining ratings into {args.output}")
    combine_ratings(bt_ratings, dg_ratings, args.output)


if __name__ == '__main__':
    main()
