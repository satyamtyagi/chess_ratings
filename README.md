# Chess Ratings Project

This project implements and compares different chess rating systems: Bradley-Terry and ELO.

## Components

- `generate_games.py`: Generates random chess game results between players
- `bradley_terry.py`: Implements the Bradley-Terry rating algorithm
- `elo_ratings.py`: Implements the ELO rating algorithm with sequential processing
- `compare_ratings.py`: Compares the two rating systems by calculating deviations

## Usage

### Generate Random Games
```bash
python generate_games.py -p 10 -g 100
```
This generates a CSV file with random game results between 10 players, with 100 total games.

### Calculate Bradley-Terry Ratings
```bash
python bradley_terry.py
```
This reads the games from `games.csv` and outputs Bradley-Terry ratings to `ratings.csv`.

### Calculate ELO Ratings
```bash
python elo_ratings.py
```
This reads the games from `games.csv` and outputs ELO ratings to `elo_ratings.csv`.

### Compare Rating Systems
```bash
python compare_ratings.py
```
This compares the ratings from both systems and calculates deviation metrics.
