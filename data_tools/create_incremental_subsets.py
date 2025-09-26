#!/usr/bin/env python3
"""
Create incremental subsets from a BT games CSV file.
Takes a dataset and creates N incremental subsets of evenly distributed sizes,
automatically determining the sizes based on the total number of games.
"""

import argparse
import csv
import os
from pathlib import Path


def create_incremental_subsets(input_file, output_dir="datasets", base_name=None, num_subsets=10):
    """
    Create incremental subsets from input CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save output files
        base_name (str): Base name for output files (auto-detected if None)
        num_subsets (int): Number of incremental subsets to create
    """
    
    # Auto-detect base name if not provided
    if base_name is None:
        base_name = Path(input_file).stem
        if base_name.endswith('_games'):
            base_name = base_name[:-6]  # Remove '_games' suffix
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input file
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        games = list(reader)   # Read all games
    
    total_games = len(games)
    print(f"Total games in input: {total_games}")
    
    # Calculate subset sizes based on total games and number of subsets
    if total_games < num_subsets:
        print(f"Error: Input file has only {total_games} games, but requested {num_subsets} subsets")
        return
    
    # Create evenly spaced subset sizes
    step_size = total_games // num_subsets
    subset_sizes = [step_size * (i + 1) for i in range(num_subsets)]
    
    # Ensure the last subset includes all games
    subset_sizes[-1] = total_games
    
    print(f"Creating {len(subset_sizes)} incremental subsets:")
    
    # Create each subset
    for i, size in enumerate(subset_sizes):
        # Create output filename with appropriate suffix
        if size >= 1000:
            suffix = f"{size//1000}k"
        else:
            suffix = f"{size}"
        output_file = os.path.join(output_dir, f"{base_name}_{suffix}_games.csv")
        
        # Write subset to file
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # Write header
            writer.writerows(games[:size])  # Write first 'size' games
        
        print(f"  Created: {output_file} ({size} games)")
    
    print(f"\nSuccessfully created {len(subset_sizes)} incremental subset files!")


def main():
    parser = argparse.ArgumentParser(
        description='Create incremental subsets from BT games CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_incremental_subsets.py bt_20k_games.csv
  python create_incremental_subsets.py bt_200k_20players_games.csv --base-name bt_200k_20p --num-subsets 15
  python create_incremental_subsets.py large_dataset.csv --output-dir subsets --num-subsets 5
        """
    )
    
    parser.add_argument('input_file', type=str,
                       help='Input CSV file with BT games')
    parser.add_argument('--output-dir', type=str, default='datasets',
                       help='Output directory for subset files (default: datasets)')
    parser.add_argument('--base-name', type=str, default=None,
                       help='Base name for output files (auto-detected from input if not provided)')
    parser.add_argument('--num-subsets', type=int, default=10,
                       help='Number of incremental subsets to create (default: 10)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return 1
    
    # Create incremental subsets
    try:
        create_incremental_subsets(
            input_file=args.input_file,
            output_dir=args.output_dir,
            base_name=args.base_name,
            num_subsets=args.num_subsets
        )
    except Exception as e:
        print(f"Error creating subsets: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
