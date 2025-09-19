#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parquet file reader script
This script demonstrates how to open, read and analyze Parquet files using pandas and pyarrow
Optimized for large files with memory-efficient options
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import os
import argparse
import gc


def get_parquet_metadata(file_path):
    """Get metadata about a parquet file without loading its contents"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        
        # Read metadata only
        metadata = pq.read_metadata(file_path)
        schema = pq.read_schema(file_path)
        
        print("\n===== Parquet File Metadata =====")
        print(f"File path: {file_path}")
        print(f"Number of rows: {metadata.num_rows}")
        print(f"Number of columns: {len(schema.names)}")
        print(f"Number of row groups: {metadata.num_row_groups}")
        print(f"File size: {metadata.serialized_size / (1024 * 1024):.2f} MB")
        print(f"Created by: {metadata.created_by if metadata.created_by else 'Unknown'}")
        
        print("\nSchema:")
        for i, field in enumerate(schema):
            print(f"  {i+1}. {field.name}: {field.type}")
            
        return metadata, schema
    except Exception as e:
        print(f"Error reading parquet metadata: {e}")
        return None, None

def read_parquet_sample(file_path, num_rows=5, columns=None):
    """Read just a sample of rows from a parquet file"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None
        
        # Read just the first few rows
        df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
        return df.head(num_rows)
    
    except Exception as e:
        print(f"Error reading parquet sample: {e}")
        return None


def display_sample_info(df):
    """Display basic information about a DataFrame sample"""
    if df is None:
        return
    
    print("\n===== Sample Data =====")
    print(f"Sample shape: {df.shape} (rows, columns)")
    
    print(f"\nColumns: {', '.join(df.columns)}")
    
    print("\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    print("\nFirst few rows:")
    print(df)
    
    # Only calculate statistics on numeric columns for the sample
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        print("\nNumeric column statistics (sample only):")
        print(df[numeric_cols].describe().to_string())
    
    # Count nulls
    null_counts = df.isna().sum()
    if null_counts.sum() > 0:
        print("\nNull values in sample:")
        print(null_counts[null_counts > 0])


def main():
    parser = argparse.ArgumentParser(description='Read and explore a Parquet file safely')
    parser.add_argument('file_path', help='Path to the Parquet file')
    parser.add_argument('--mode', choices=['metadata', 'sample', 'count'], default='metadata',
                      help='Mode of operation: metadata (default, safest), sample (read first N rows), count (column statistics)')
    parser.add_argument('--columns', help='Comma-separated list of columns to read (optional)')
    parser.add_argument('--rows', type=int, default=5, help='Number of rows to read in sample mode (default: 5)')
    parser.add_argument('--numeric-only', action='store_true', help='Only analyze numeric columns (for count mode)')
    
    args = parser.parse_args()
    
    # Parse columns if provided
    columns = None
    if args.columns:
        columns = [col.strip() for col in args.columns.split(',')]
        print(f"Using columns: {columns}")
    
    # Always get metadata first (this is fast and safe)
    metadata, schema = get_parquet_metadata(args.file_path)
    if metadata is None:
        return
        
    # Based on the mode, perform different operations
    if args.mode == 'metadata':
        print("\nFor more detailed analysis, try:")
        print(f"  python {sys.argv[0]} {args.file_path} --mode sample --rows 10")
        print(f"  python {sys.argv[0]} {args.file_path} --mode count --columns=column1,column2")
        
    elif args.mode == 'sample':
        print(f"\nReading sample of {args.rows} rows...")
        df_sample = read_parquet_sample(args.file_path, num_rows=args.rows, columns=columns)
        display_sample_info(df_sample)
        
    elif args.mode == 'count':
        try:
            # This is safer than loading everything - we'll count values column by column
            print("\nCalculating column statistics (this might take a moment)...")
            
            if columns is None:
                columns = schema.names
            
            for col in columns:
                try:
                    # Read just this column
                    print(f"\nAnalyzing column: {col}")
                    col_df = pd.read_parquet(args.file_path, columns=[col])
                    
                    # Basic stats that won't crash on large text columns
                    print(f"  Number of rows: {len(col_df)}")
                    print(f"  Number of unique values: {col_df[col].nunique()}")
                    print(f"  Number of null values: {col_df[col].isna().sum()}")
                    
                    # If numeric and numeric_only flag is set, or not numeric_only flag
                    if not args.numeric_only or pd.api.types.is_numeric_dtype(col_df[col]):
                        if pd.api.types.is_numeric_dtype(col_df[col]):
                            print("  Numeric statistics:")
                            stats = col_df[col].describe()
                            for stat_name, value in stats.items():
                                print(f"    {stat_name}: {value}")
                        elif pd.api.types.is_string_dtype(col_df[col]):
                            # For string columns, show a few examples instead of statistics
                            non_null = col_df[col].dropna()
                            if len(non_null) > 0:
                                print("  Example values (first 3):")
                                for i, val in enumerate(non_null.head(3).values):
                                    # Truncate very long strings
                                    if isinstance(val, str) and len(val) > 100:
                                        val = val[:100] + "..."
                                    print(f"    {i+1}: {val}")
                    
                    # Force garbage collection to prevent memory buildup
                    del col_df
                    gc.collect()
                    
                except Exception as e:
                    print(f"  Error analyzing column {col}: {e}")
        
        except Exception as e:
            print(f"Error analyzing columns: {e}")
            
    print("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
