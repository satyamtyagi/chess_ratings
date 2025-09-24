#!/usr/bin/env python3
"""
Generate chess games with realistic ELO-based outcomes.
Creates players with normally distributed ratings and generates games where outcomes
are determined by ELO probability calculations.
"""

import argparse
import csv
import random
import math
from pathlib import Path
import numpy as np


def generate_normal_ratings(num_players, min_rating=400, max_rating=2400, mean_rating=1400):
    """
    Generate normally distributed player ratings.
    
    Args:
        num_players (int): Number of players
        min_rating (int): Minimum allowed rating
        max_rating (int): Maximum allowed rating  
        mean_rating (int): Mean of the distribution
        
    Returns:
        list: List of ratings for each player (clipped to min/max bounds)
    """
    # Calculate standard deviation to get good spread within bounds
    # Using rule of thumb: 99.7% of values within 3 standard deviations
    std_deviation = (max_rating - min_rating) / 6
    
    ratings = []
    for _ in range(num_players):
        rating = np.random.normal(mean_rating, std_deviation)
        # Clip to bounds
        rating = max(min_rating, min(max_rating, rating))
        ratings.append(int(round(rating)))
    
    return ratings


def elo_win_probability(rating1, rating2):
    """
    Calculate probability that player 1 beats player 2 using ELO formula.
    
    Args:
        rating1 (int): Player 1's rating
        rating2 (int): Player 2's rating
        
    Returns:
        float: Probability that player 1 wins (0.0 to 1.0)
    """
    # Standard ELO expected score formula
    # P(Player 1 wins) = 1 / (1 + 10^((R2 - R1)/400))
    return 1.0 / (1.0 + 10**((rating2 - rating1) / 400.0))


def generate_rating_based_games(num_players, num_games, ratings):
    """
    Generate games where outcomes are determined by ELO probabilities.
    
    Args:
        num_players (int): Number of players
        num_games (int): Number of games to generate
        ratings (list): List of player ratings
        
    Returns:
        list: List of game tuples (game_no, player1, player2, result)
    """
    games = []
    
    for game_no in range(1, num_games + 1):
        # Randomly select two different players
        player1 = random.randint(1, num_players)
        player2 = random.randint(1, num_players)
        while player2 == player1:
            player2 = random.randint(1, num_players)
        
        # Get their ratings (players are 1-indexed, ratings are 0-indexed)
        rating1 = ratings[player1 - 1]
        rating2 = ratings[player2 - 1]
        
        # Calculate win probability for player 1
        win_prob = elo_win_probability(rating1, rating2)
        
        # Determine result based on probability
        if random.random() < win_prob:
            result = 'w'  # Player 1 wins
        else:
            result = 'l'  # Player 1 loses
        
        games.append((game_no, player1, player2, result))
    
    return games


def save_ratings_csv(ratings, filename):
    """Save player ratings to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['player_id', 'rating'])
        
        for player_id, rating in enumerate(ratings, 1):
            writer.writerow([player_id, rating])


def save_games_csv(games, filename):
    """Save games to CSV file in the same format as generate_games.py."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['game_no', 'player_first', 'player_second', 'result'])
        
        for game in games:
            writer.writerow(game)


def main():
    parser = argparse.ArgumentParser(description='Generate chess games with ELO-based realistic outcomes')
    parser.add_argument('-p', '--players', type=int, default=20,
                       help='Number of players (default: 20)')
    parser.add_argument('-g', '--games', type=int, default=2000,
                       help='Number of games to generate (default: 2000)')
    parser.add_argument('--min-rating', type=int, default=400,
                       help='Minimum player rating (default: 400)')
    parser.add_argument('--max-rating', type=int, default=2400,
                       help='Maximum player rating (default: 2400)')
    parser.add_argument('--mean-rating', type=int, default=1400,
                       help='Mean player rating (default: 1400)')
    parser.add_argument('--output-games', type=str, default='datasets/normal_games.csv',
                       help='Output filename for games CSV (default: datasets/normal_games.csv)')
    parser.add_argument('--output-ratings', type=str, default='datasets/player_ratings.csv',
                       help='Output filename for ratings CSV (default: datasets/player_ratings.csv)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    print(f"Generating {args.players} players with ratings...")
    ratings = generate_normal_ratings(args.players, args.min_rating, args.max_rating, args.mean_rating)
    
    print(f"Player ratings: min={min(ratings)}, max={max(ratings)}, mean={sum(ratings)/len(ratings):.1f}")
    
    print(f"Generating {args.games} games with ELO-based outcomes...")
    games = generate_rating_based_games(args.players, args.games, ratings)
    
    # Save both files
    save_ratings_csv(ratings, args.output_ratings)
    save_games_csv(games, args.output_games)
    
    print(f"Generated files:")
    print(f"  Games: {args.output_games}")
    print(f"  Ratings: {args.output_ratings}")
    
    # Show some statistics
    wins = sum(1 for game in games if game[3] == 'w')
    losses = len(games) - wins
    print(f"Results: {wins} wins, {losses} losses for player_first")


if __name__ == '__main__':
    main()
