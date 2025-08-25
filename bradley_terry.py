#!/usr/bin/env python3
# Counter for expected score calculations
expected_score_calls = 0
"""
Bradley-Terry Rating Algorithm Implementation

This script implements the Bradley-Terry model for estimating player ratings based on game outcomes.
Players start with a rating of 0, and the algorithm iterates until convergence (error < threshold).

The Bradley-Terry model estimates the probability that player i beats player j as:
P(i beats j) = exp(r_i) / (exp(r_i) + exp(r_j))
where r_i and r_j are the ratings of players i and j.
"""

import argparse
import csv
import math
import numpy as np
from collections import defaultdict
import os


def load_games_from_csv(file_path):
    """
    Load game data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing game data
        
    Returns:
        list: List of dictionaries containing game data
    """
    games = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            game = {
                'player_first': int(row['player_first']),
                'player_second': int(row['player_second']),
                'result': row['result']
            }
            games.append(game)
    
    return games


def get_player_list(games):
    """
    Get a list of unique player IDs from game data.
    
    Args:
        games (list): List of dictionaries containing game data
        
    Returns:
        list: List of unique player IDs
    """
    player_set = set()
    for game in games:
        player_set.add(game['player_first'])
        player_set.add(game['player_second'])
    
    return sorted(list(player_set))


def initialize_ratings(players, initial_rating=0):
    """
    Initialize ratings for all players.
    
    Args:
        players (list): List of player IDs
        initial_rating (float): Initial rating for all players
        
    Returns:
        dict: Dictionary mapping player IDs to their initial ratings
    """
    return {player: initial_rating for player in players}


def compute_expected_score(player1_rating, player2_rating):
    """
    Compute the expected score for player1 against player2 using Bradley-Terry model.
    
    Args:
        player1_rating (float): Rating of player 1
        player2_rating (float): Rating of player 2
        
    Returns:
        float: Expected score (probability of player1 winning)
    """
    global expected_score_calls
    expected_score_calls += 1
    return 1.0 / (1.0 + math.exp(player2_rating - player1_rating))


def compute_expected_scores(ratings, games):
    """
    Compute expected scores for all games based on current ratings.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their ratings
        games (list): List of dictionaries containing game data
        
    Returns:
        dict: Dictionary mapping player IDs to their expected scores
    """
    expected_scores = defaultdict(float)
    
    for game in games:
        player1 = game['player_first']
        player2 = game['player_second']
        
        expected_score = compute_expected_score(ratings[player1], ratings[player2])
        
        expected_scores[player1] += expected_score
        expected_scores[player2] += (1 - expected_score)
    
    return expected_scores


def compute_actual_scores(games):
    """
    Compute actual scores for all players based on game outcomes.
    
    Args:
        games (list): List of dictionaries containing game data
        
    Returns:
        dict: Dictionary mapping player IDs to their actual scores
    """
    actual_scores = defaultdict(float)
    
    for game in games:
        player1 = game['player_first']
        player2 = game['player_second']
        result = game['result']
        
        if result == 'w':
            actual_scores[player1] += 1
        else:
            actual_scores[player2] += 1
    
    return actual_scores


def update_ratings(ratings, actual_scores, expected_scores, learning_rate=1.0, normalize=True):
    """
    Update ratings based on actual and expected scores.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their current ratings
        actual_scores (dict): Dictionary mapping player IDs to their actual scores
        expected_scores (dict): Dictionary mapping player IDs to their expected scores
        learning_rate (float): Learning rate for updating ratings
        normalize (bool): Whether to normalize ratings by subtracting the mean
        
    Returns:
        dict: Dictionary mapping player IDs to their updated ratings
        float: Maximum absolute error between expected and actual scores
    """
    new_ratings = {}
    max_error = 0
    
    for player in ratings:
        if player in actual_scores:
            error = actual_scores[player] - expected_scores[player]
            max_error = max(max_error, abs(error))
            new_ratings[player] = ratings[player] + learning_rate * error
        else:
            new_ratings[player] = ratings[player]
    
    # Normalize ratings by subtracting the mean if requested
    if normalize:
        mean_rating = sum(new_ratings.values()) / len(new_ratings)
        for player in new_ratings:
            new_ratings[player] -= mean_rating
    
    return new_ratings, max_error


def convert_to_elo(bt_rating, elo_anchor=1200, bt_anchor=0):
    """
    Convert Bradley-Terry rating to ELO scale.
    
    Args:
        bt_rating (float): Bradley-Terry rating
        elo_anchor (int): ELO rating anchor point (default: 1200)
        bt_anchor (float): Bradley-Terry rating anchor point (default: 0)
        
    Returns:
        float: Corresponding ELO rating
    """
    # Formula: ELO = elo_anchor + (400/ln(10)) * (BT_rating - bt_anchor)
    return elo_anchor + (400 / math.log(10)) * (bt_rating - bt_anchor)


def save_ratings_to_csv(ratings, wins, output_file='ratings.csv'):
    """
    Save player ratings and win counts to a CSV file.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their Bradley-Terry ratings
        wins (dict): Dictionary mapping player IDs to their win counts
        output_file (str): Name of the output CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['player_no', 'bt_rating', 'elo_rating', 'wins'])
        
        # Write ratings and wins for each player
        for player, bt_rating in sorted(ratings.items()):
            elo_rating = convert_to_elo(bt_rating)
            win_count = wins.get(player, 0)
            writer.writerow([player, f"{bt_rating:.6f}", f"{elo_rating:.1f}", win_count])
            
    print(f"Successfully saved ratings to {output_file}")


def bradley_terry_batch(games, max_iterations=1000, threshold=0.0001, learning_rate=1.0, normalize=True):
    """
    Run Bradley-Terry rating algorithm on a batch of games.
    
    Args:
        games (list): List of dictionaries containing game data
        max_iterations (int): Maximum number of iterations
        threshold (float): Convergence threshold for error
        learning_rate (float): Learning rate for updating ratings
        
    Returns:
        dict: Dictionary mapping player IDs to their final ratings
        dict: Dictionary mapping player IDs to their win counts
        int: Number of iterations performed
        float: Final error
    """
    players = get_player_list(games)
    ratings = initialize_ratings(players)
    
    for iteration in range(max_iterations):
        expected_scores = compute_expected_scores(ratings, games)
        actual_scores = compute_actual_scores(games)
        
        new_ratings, error = update_ratings(ratings, actual_scores, expected_scores, learning_rate, normalize)
        
        print(f"Iteration {iteration + 1}: Maximum Error = {error:.6f}, Expected score calls so far: {expected_score_calls}")
        
        if error < threshold:
            print(f"Converged after {iteration + 1} iterations!")
            print(f"Total expected score calculations: {expected_score_calls}")
            return new_ratings, actual_scores, iteration+1, error
        
        ratings = new_ratings
    
    print(f"Did not converge after {max_iterations} iterations. Final error: {error:.6f}")
    return ratings, actual_scores, max_iterations, error


def display_ratings(ratings, wins):
    """
    Display player ratings and win counts in descending order by rating.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their Bradley-Terry ratings
        wins (dict): Dictionary mapping player IDs to their win counts
    """
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFinal Ratings:")
    print("-" * 60)
    print(f"{'Rank':4} | {'Player':6} | {'BT Rating':12} | {'ELO Rating':10} | {'Wins':5}")
    print("-" * 60)
    
    for rank, (player, bt_rating) in enumerate(sorted_ratings, 1):
        elo_rating = convert_to_elo(bt_rating)
        win_count = wins.get(player, 0)
        print(f"{rank:4} | {player:6} | {bt_rating:12.6f} | {elo_rating:10.1f} | {win_count:5.0f}")


def main():
    global expected_score_calls
    expected_score_calls = 0
    
    parser = argparse.ArgumentParser(description='Run Bradley-Terry algorithm on game data.')
    parser.add_argument('-f', '--file', type=str, default='games.csv',
                        help='CSV file containing game data (default: games.csv)')
    parser.add_argument('-t', '--threshold', type=float, default=0.0001,
                        help='Convergence threshold (default: 0.0001)')
    parser.add_argument('-i', '--iterations', type=int, default=1000,
                        help='Maximum number of iterations (default: 1000)')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1,
                        help='Learning rate for rating updates (default: 0.1)')
    parser.add_argument('-n', '--no-normalize', action='store_true',
                        help='Disable normalization of ratings after each iteration')
    parser.add_argument('-o', '--output-csv', type=str, default='ratings.csv',
                        help='Output CSV file for ratings (default: ratings.csv)')
    
    args = parser.parse_args()
    
    print(f"Loading games from {args.file}...")
    games = load_games_from_csv(args.file)
    print(f"Loaded {len(games)} games involving {len(get_player_list(games))} players")
    
    print(f"\nRunning Bradley-Terry algorithm...")
    print(f"Max iterations: {args.iterations}, Threshold: {args.threshold}, Learning rate: {args.learning_rate}")
    print(f"Normalization: {not args.no_normalize}")
    
    ratings, wins, iterations, error = bradley_terry_batch(
        games, 
        max_iterations=args.iterations,
        threshold=args.threshold,
        learning_rate=args.learning_rate,
        normalize=not args.no_normalize
    )
    
    # Save ratings to CSV
    save_ratings_to_csv(ratings, wins, args.output_csv)
    
    display_ratings(ratings, wins)
    
    # Calculate and display some statistics
    rating_values = list(ratings.values())
    print("\nRating Statistics:")
    print(f"Mean Rating: {np.mean(rating_values):.6f}")
    print(f"Median Rating: {np.median(rating_values):.6f}")
    print(f"Min Rating: {min(rating_values):.6f}")
    print(f"Max Rating: {max(rating_values):.6f}")
    print(f"Standard Deviation: {np.std(rating_values):.6f}")


if __name__ == '__main__':
    main()
