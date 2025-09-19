#!/usr/bin/env python3
"""
Game Data Shuffler

This script creates multiple shuffled versions of a game data CSV file.
It preserves the header row if present and only shuffles the game data rows.
"""

import argparse
import csv
import os
import random


def read_game_data(input_file, has_header=True):
    """
    Read game data from CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        has_header (bool): Whether the CSV file has a header row
        
    Returns:
        tuple: (header_row, game_data_rows)
    """
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Get the header row if present
        header_row = next(reader) if has_header else None
        
        # Read all game data rows
        game_data = list(reader)
        
    return header_row, game_data


def write_shuffled_data(output_file, header_row, game_data):
    """
    Write shuffled game data to a new CSV file.
    
    Args:
        output_file (str): Path to the output CSV file
        header_row (list): Header row (can be None)
        game_data (list): List of game data rows to shuffle and write
    """
    # Create a shuffled copy of the game data
    shuffled_data = game_data.copy()
    random.shuffle(shuffled_data)
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row if present
        if header_row:
            writer.writerow(header_row)
        
        # Write shuffled game data
        for row in shuffled_data:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Generate shuffled versions of a game data CSV file.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input CSV file')
    parser.add_argument('--output-dir', '-o', default='shuffled', help='Directory for output files')
    parser.add_argument('--num-shuffles', '-n', type=int, default=10, help='Number of shuffled versions to generate')
    parser.add_argument('--has-header', action='store_true', help='Whether the input CSV has a header row')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--prefix', default='shuffle_', help='Prefix for output filenames')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the input game data
    header_row, game_data = read_game_data(args.input, args.has_header)
    
    # Get base input filename without extension
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    
    # Generate shuffled versions
    for i in range(1, args.num_shuffles + 1):
        # Create output filename
        output_file = os.path.join(args.output_dir, 
                                  f"{args.prefix}{input_basename}_{i:02d}.csv")
        
        # Write shuffled data
        write_shuffled_data(output_file, header_row, game_data)
        print(f"Created shuffle {i}/{args.num_shuffles}: {output_file}")


if __name__ == '__main__':
    main()
