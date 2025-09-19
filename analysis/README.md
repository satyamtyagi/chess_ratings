# Analysis Tools

This directory contains utilities for comparing, analyzing, and visualizing rating algorithm performance.

## Files

### Performance Analysis
- **`compare_ratings.py`** - Compares different rating systems (Bradley-Terry vs ELO) by calculating MSE, deviations, and prediction accuracy
- **`calculate_shuffle_rmse.py`** - Calculates RMSE between rating algorithms when game order is shuffled, tests robustness
- **`plot_rmse.py`** - Generates plots and visualizations of RMSE comparisons between algorithms

### Data Processing & Utilities  
- **`combine_ratings.py`** - Merges multiple rating files into a single combined dataset
- **`create_game_subsets.py`** - Creates subsets of game datasets for testing with different sample sizes
- **`dirty_graph_adapter.py`** - Adapter/wrapper utilities for integrating Dirty Graph algorithm with other systems

## Usage Examples

```bash
# Compare Bradley-Terry and ELO ratings
python analysis/compare_ratings.py results/bt_ratings.csv results/elo_ratings.csv

# Calculate RMSE after shuffling games  
python analysis/calculate_shuffle_rmse.py datasets/games.csv --shuffles 10

# Plot RMSE comparison across algorithms
python analysis/plot_rmse.py results/ --output analysis_plots/

# Combine multiple rating files
python analysis/combine_ratings.py results/bt_*.csv -o results/combined_ratings.csv

# Create game subsets for testing
python analysis/create_game_subsets.py datasets/huge_games.csv --sizes 500,1000,2000
```

## Input/Output

- **Input**: Rating files from `results/`, game datasets from `datasets/`
- **Output**: 
  - Comparison reports and statistics
  - Plots and visualizations  
  - Combined/processed rating files
  - Performance metrics (RMSE, correlation, accuracy)

## Dependencies

Most tools require:
- pandas
- numpy  
- matplotlib (for plotting)
- scipy (for statistical functions)
