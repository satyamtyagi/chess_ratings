# Experiments

This directory contains various rating algorithm implementations and experimental versions used during development and research.

## Files

### Core Bradley-Terry Implementations
- **`bradley_terry.py`** - Main Bradley-Terry implementation
- **`bradley_terry_diag_newton.py`** - Diagonal Newton method for Bradley-Terry
- **`bradley_terry_mm.py`** - Majorization-Minimization approach for Bradley-Terry

### Single-Pass Algorithm Baselines
- **`bt_single_pass_baselines.py`** - Original single-pass baselines
- **`bt_single_pass_baselines_opdn_fixed.py`** - Fixed OPDN implementation
- **`bt_single_pass_baselines_opdn_v2.py`** - OPDN version 2
- **`bt_single_pass_baselines_opdn_v2_fixed.py`** - Fixed OPDN version 2
- **`bt_single_pass_baselines_opdn_v3.py`** - OPDN version 3
- **`bt_single_pass_baselines_opdn_v3_fixed.py`** - Fixed OPDN version 3

### Dirty Graph Algorithm Variants
- **`dirty_graph_ratings.py`** - Original Dirty Graph implementation
- **`dirty_graph_ratings_v2.py`** - Dirty Graph version 2
- **`dirty_graph_ratings_v2_fixed.py`** - Fixed Dirty Graph version 2
- **`dirty_graph_ratings_nolr3.py`** - Dirty Graph without learning rate 3
- **`dirty_graph_ratings_nolr4.py`** - Dirty Graph without learning rate 4
- **`dirty_graph_ratings_nolr5.py`** - Dirty Graph without learning rate 5
- **`Copy of dirty_graph_ratings.py`** - Backup copy of original implementation

### Alternative Rating Systems
- **`elo_ratings.py`** - ELO rating system implementation
- **`trueskill_rater.py`** - TrueSkill rating system implementation

## Current Main Implementation

The current main implementation is in the root directory:
- **`../bt_single_pass_with_dg.py`** - Production-ready implementation with Dirty Graph and all single-pass algorithms

## Usage

These files represent the evolution of the rating algorithms and can be used for:
- Historical reference
- Comparative studies
- Algorithm variant testing
- Research benchmarking

Most experimental files follow similar command-line interfaces:

```bash
# Example usage for experimental files
python experiments/bradley_terry.py --csv datasets/games.csv
python experiments/dirty_graph_ratings.py --csv datasets/games.csv --learning-rate 0.01
python experiments/elo_ratings.py --csv datasets/games.csv
```

## Note

These are experimental/developmental versions. For production use, refer to the main implementation in the root directory.
