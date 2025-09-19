#!/usr/bin/env python3
"""
Game Subset Generator

This script creates 10 subsets of a CSV game file with different numbers of games.
Each subset contains games from 1 to N, where N is the subset size.
"""

import csv
import os
import argparse

def create_game_subsets(input_file, num_subsets=10):
    """
    Create subsets of the input CSV file with different numbers of games.
    
    Args:
        input_file (str): Path to the input CSV file
        num_subsets (int): Number of subsets to create
    """
    # Read the input file to determine total number of games
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Save the header
        games = list(reader)
    
    total_games = len(games)
    print(f"Total games in {input_file}: {total_games}")
    
    # Calculate subset sizes
    subset_sizes = [total_games // num_subsets * i for i in range(1, num_subsets + 1)]
    # Make sure the last subset includes all games
    subset_sizes[-1] = total_games
    
    # Create each subset file
    for size in subset_sizes:
        # Create output filename based on size
        output_file = f"1_{size}_huge_games.csv"
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write the header
            
            # Write games from 1 to size
            for i in range(min(size, total_games)):
                writer.writerow(games[i])
        
        print(f"Created {output_file} with {min(size, total_games)} games")

def main():
    parser = argparse.ArgumentParser(description='Create subsets of game data CSV file.')
    parser.add_argument('-f', '--file', type=str, default='huge_games.csv',
                        help='Input CSV file containing game data (default: huge_games.csv)')
    parser.add_argument('-n', '--num-subsets', type=int, default=10,
                        help='Number of subsets to create (default: 10)')
    
    args = parser.parse_args()
    
    create_game_subsets(args.file, num_subsets=args.num_subsets)
    print(f"All subset files created successfully from {args.file}!")

if __name__ == "__main__":
    main()
