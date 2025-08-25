# Chess Ratings Project

This project implements and compares different chess rating systems: Bradley-Terry, ELO, and Dirty Graph ratings.

## Components

- `generate_games.py`: Generates random chess game results between players
- `bradley_terry.py`: Implements the Bradley-Terry rating algorithm with win statistics tracking
- `elo_ratings.py`: Implements the ELO rating algorithm with sequential processing and win statistics tracking
- `dirty_graph_ratings.py`: Implements the Dirty Graph rating algorithm with edge-based updates and win statistics tracking
- `compare_ratings.py`: Compares different rating systems by calculating deviations and displaying win statistics

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
This reads the games from `games.csv` and outputs ELO ratings to `elo_ratings.csv`. The output includes player win statistics.

### Calculate Dirty Graph Ratings
```bash
python dirty_graph_ratings.py
```
This reads the games from `games.csv` and outputs Dirty Graph ratings to `dirty_graph_ratings.csv`. It uses an edge-based incremental update approach with a default learning rate of 1.0.

### Compare Rating Systems
```bash
python compare_ratings.py -f1 ratings.csv -f2 elo_ratings.csv
```
This compares the ratings from two systems and calculates deviation metrics including MSE, RMSE, and correlations. It also displays win statistics from the first file.

You can compare any two rating systems by specifying different file pairs:
```bash
python compare_ratings.py -f1 ratings.csv -f2 dirty_graph_ratings.csv
python compare_ratings.py -f1 elo_ratings.csv -f2 dirty_graph_ratings.csv
```

## Win Statistics

All three rating algorithms now track win statistics for each player. This feature has been implemented to provide additional performance metrics alongside the computed ratings.

### Output Format

Each rating CSV file includes the following columns:
- `player_no`: Player identifier (integer)
- `bt_rating`: Bradley-Terry rating score
- `elo_rating`: ELO rating score
- `wins`: Total number of wins for the player

### Console Output

When running any of the rating algorithms, the console output includes win counts alongside the ratings:

```
Rank | Player | BT Rating    | ELO Rating | Wins 
------------------------------------------------------------
   1 |      4 |     0.752859 |     1330.8 |    30
   2 |     10 |     0.251372 |     1243.7 |    19
   ...
```

When comparing rating systems, win statistics from the first file are included in the player-by-player comparison table.
