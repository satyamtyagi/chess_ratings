# Data Tools

This directory contains utilities for data generation, modification, and parsing.

## Files

### Data Generation
- **`generate_games.py`** - Generates chess match datasets with various player counts and game numbers
- **`generate_swiss_games.py`** - Generates Swiss tournament format chess matches with proper pairing algorithms

### Data Processing  
- **`shuffle_games.py`** - Randomizes the order of games in datasets for testing algorithm robustness
- **`parquet_to_csv.py`** - Converts Parquet files to CSV format with options to filter ties and format for rating algorithms

### Data Analysis
- **`parquet_reader.py`** - Memory-efficient Parquet file analysis tool with multiple modes:
  - `metadata`: Show file structure and schema (default)
  - `sample`: Read first N rows 
  - `count`: Column-wise statistics and analysis

## Usage Examples

```bash
# Generate 1000 games between 50 players
python data_tools/generate_games.py --players 50 --games 1000 --output datasets/my_games.csv

# Convert AI model parquet to CSV without ties
python data_tools/parquet_to_csv.py datasets/train-ai.parquet -o datasets/train-ai-clean.csv

# Analyze parquet file structure
python data_tools/parquet_reader.py datasets/train-ai.parquet

# Sample first 100 rows from parquet
python data_tools/parquet_reader.py datasets/train-ai.parquet --mode sample --rows 100

# Shuffle existing games for robustness testing
python data_tools/shuffle_games.py datasets/games.csv -o datasets/games_shuffled.csv
```

## Input/Output

- **Input**: Raw data files, existing datasets
- **Output**: Processed CSV files in `datasets/` directory
- **Results**: Rating algorithm outputs go to `results/` directory
