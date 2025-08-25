#!/usr/bin/env python3
"""
Standard ELO Rating Implementation

This script implements the standard ELO rating system for chess games. 
It processes game data from a CSV file and calculates ELO ratings for all players.
It also provides conversion from ELO to Bradley-Terry ratings.

The ELO model uses the following formula to update ratings:
New Rating = Old Rating + K * (Actual Score - Expected Score)

where K is a factor determining the maximum rating change per game,
and Expected Score = 1 / (1 + 10^((opponent_rating - player_rating) / 400))
"""

import argparse
import csv
import math
import numpy as np


def elo_to_bradley_terry(elo_rating, elo_anchor=1200, bt_anchor=0):
    """
    Convert ELO rating to Bradley-Terry scale.
    
    Args:
        elo_rating (float): ELO rating
        elo_anchor (int): ELO rating anchor point (default: 1200)
        bt_anchor (float): Bradley-Terry rating anchor point (default: 0)
        
    Returns:
        float: Corresponding Bradley-Terry rating
    """
    # Formula: BT = bt_anchor + (elo_rating - elo_anchor) * ln(10) / 400
    return bt_anchor + (elo_rating - elo_anchor) * math.log(10) / 400


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
                'game_no': int(row['game_no']),
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


def initialize_ratings(players, initial_rating=1200):
    """
    Initialize ELO ratings for all players.
    
    Args:
        players (list): List of player IDs
        initial_rating (int): Initial ELO rating for all players (default: 1200)
        
    Returns:
        dict: Dictionary mapping player IDs to their initial ratings
    """
    return {player: initial_rating for player in players}


def compute_expected_score_elo(player_rating, opponent_rating):
    """
    Compute the expected score for a player against an opponent using ELO formula.
    
    Args:
        player_rating (float): ELO rating of the player
        opponent_rating (float): ELO rating of the opponent
        
    Returns:
        float: Expected score (probability of player winning)
    """
    return 1.0 / (1.0 + math.pow(10, (opponent_rating - player_rating) / 400))


def update_elo_ratings(ratings, games, k_factor=32):
    """
    Update ELO ratings based on game outcomes.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their current ELO ratings
        games (list): List of dictionaries containing game data
        k_factor (int): Maximum rating change per game
        
    Returns:
        dict: Dictionary mapping player IDs to their updated ELO ratings
        float: Average absolute rating change
    """
    new_ratings = ratings.copy()
    total_change = 0
    
    for game in games:
        player1 = game['player_first']
        player2 = game['player_second']
        result = game['result']
        
        # Determine actual score
        if result == 'w':  # player1 wins
            score1 = 1.0
            score2 = 0.0
        else:  # player2 wins
            score1 = 0.0
            score2 = 1.0
        
        # Compute expected scores using continuously updated ratings (new_ratings)
        # for sequential processing
        expected1 = compute_expected_score_elo(new_ratings[player1], new_ratings[player2])
        expected2 = compute_expected_score_elo(new_ratings[player2], new_ratings[player1])
        
        # Update ratings
        rating_change1 = k_factor * (score1 - expected1)
        rating_change2 = k_factor * (score2 - expected2)
        
        new_ratings[player1] += rating_change1
        new_ratings[player2] += rating_change2
        
        total_change += abs(rating_change1) + abs(rating_change2)
    
    average_change = total_change / (2 * len(games)) if games else 0
    return new_ratings, average_change





def display_elo_ratings(elo_ratings):
    """
    Display player ELO ratings in descending order with their equivalent BT ratings.
    
    Args:
        elo_ratings (dict): Dictionary mapping player IDs to their ELO ratings
    """
    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFinal ELO Ratings:")
    print("-" * 50)
    print(f"{'Rank':4} | {'Player':6} | {'ELO Rating':10} | {'BT Rating':12}")
    print("-" * 50)
    
    for rank, (player, elo) in enumerate(sorted_ratings, 1):
        bt = elo_to_bradley_terry(elo)
        print(f"{rank:4} | {player:6} | {elo:10.1f} | {bt:12.6f}")


def save_ratings_to_csv(elo_ratings, output_file='elo_ratings.csv'):
    """
    Save player ELO ratings to a CSV file, along with equivalent BT ratings.
    Matching the format of ratings.csv: player_no, bt_rating, elo_rating.
    
    Args:
        elo_ratings (dict): Dictionary mapping player IDs to their ELO ratings
        output_file (str): Name of the output CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header to match ratings.csv format
        writer.writerow(['player_no', 'bt_rating', 'elo_rating'])
        
        # Write ratings for each player
        for player, elo in sorted(elo_ratings.items()):
            bt = elo_to_bradley_terry(elo)
            writer.writerow([player, f"{bt:.6f}", f"{elo:.1f}"])
            
    print(f"Successfully saved ELO ratings to {output_file}")


def calculate_elo_ratings(games, k_factor=32):
    """
    Calculate ELO ratings from game data.
    
    Args:
        games (list): List of dictionaries containing game data
        k_factor (int): Maximum rating change per game
        
    Returns:
        dict: Dictionary mapping player IDs to their final ELO ratings
    """
    players = get_player_list(games)
    ratings = initialize_ratings(players)
    
    # Process all games once
    ratings, _ = update_elo_ratings(ratings, games, k_factor)
    
    return ratings





def main():
    parser = argparse.ArgumentParser(description='ELO rating calculation for chess games.')
    parser.add_argument('-f', '--file', type=str, default='games.csv',
                        help='CSV file containing game data (default: games.csv)')
    parser.add_argument('-k', '--k-factor', type=int, default=32,
                        help='K-factor for ELO updates (default: 32)')
    parser.add_argument('-o', '--output-csv', type=str, default='elo_ratings.csv',
                        help='Output CSV file for ELO ratings (default: elo_ratings.csv)')
    
    args = parser.parse_args()
    
    print(f"Loading games from {args.file}...")
    games = load_games_from_csv(args.file)
    print(f"Loaded {len(games)} games involving {len(get_player_list(games))} players")
    
    print(f"\nCalculating ELO ratings...")
    print(f"K-factor: {args.k_factor}")
    
    # Calculate ELO ratings
    elo_ratings = calculate_elo_ratings(games, k_factor=args.k_factor)
    
    # Save ELO ratings to CSV
    save_ratings_to_csv(elo_ratings, args.output_csv)
    
    # Display ratings
    display_elo_ratings(elo_ratings)
    
    # Calculate and display some statistics
    rating_values = list(elo_ratings.values())
    print("\nELO Rating Statistics:")
    print(f"Mean Rating: {np.mean(rating_values):.1f}")
    print(f"Median Rating: {np.median(rating_values):.1f}")
    print(f"Min Rating: {min(rating_values):.1f}")
    print(f"Max Rating: {max(rating_values):.1f}")
    print(f"Standard Deviation: {np.std(rating_values):.1f}")


if __name__ == '__main__':
    main()
