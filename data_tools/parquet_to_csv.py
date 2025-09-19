#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parquet to CSV converter
Extracts specific columns from a parquet file and converts to CSV format
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import argparse

def convert_to_csv(input_file, output_file=None, drop_ties=True):
    """
    Convert parquet file to CSV with specific columns:
    - Number (row index)
    - model_a
    - model_b
    - result (w if model_a won, l if model_b won)
    
    Parameters:
    - input_file: Path to parquet file
    - output_file: Path for output CSV file (default: replace .parquet with .csv)
    - drop_ties: Whether to remove ties from the dataset (default: True)
    """
    if output_file is None:
        output_file = input_file.replace('.parquet', '.csv')
    
    print(f"Reading from {input_file}...")
    
    try:
        # Only read the columns we need
        df = pd.read_parquet(input_file, columns=['model_a', 'model_b', 'winner'])
        
        # Add row number starting from 1
        df.insert(0, 'Number', range(1, len(df) + 1))
        
        # Create result column (w = model_a won, l = model_b won)
        df['result'] = df.apply(lambda row: 'w' if row['winner'] == 'model_a' else 
                                        'l' if row['winner'] == 'model_b' else None, axis=1)
        
        # Filter out ties
        original_count = len(df)
        df = df.dropna(subset=['result'])
        ties_dropped = original_count - len(df)
        print(f"Dropped {ties_dropped} ties ({(ties_dropped/original_count)*100:.1f}% of the dataset)")
        
        # Select and reorder columns
        result_df = df[['Number', 'model_a', 'model_b', 'result']]
        
        # Write to CSV
        print(f"Writing {len(result_df)} rows to {output_file}...")
        result_df.to_csv(output_file, index=False)
        
        print(f"Successfully created {output_file}")
        print("\nFirst few rows:")
        print(result_df.head())
        
        return True
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert parquet to CSV with specific columns')
    parser.add_argument('input_file', help='Path to the input parquet file')
    parser.add_argument('--output', '-o', help='Path to the output CSV file (optional)')
    
    args = parser.parse_args()
    
    convert_to_csv(args.input_file, args.output)

if __name__ == "__main__":
    main()
