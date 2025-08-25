#!/usr/bin/env python3
"""
Generate random chess game results for a specified number of players and games.
Results are saved to a CSV file with columns for game_no, player_first, player_second, and result (w/l).
"""

import argparse
import csv
import random
from pathlib import Path


def generate_random_games(num_players, num_games):
    """
    Generate random game results between players.
    
    Args:
        num_players (int): Number of players
        num_games (int): Number of games to generate
        
    Returns:
        list: List of tuples containing (player_first, player_second, result)
    """
    games = []
    
    for _ in range(num_games):
        # Select two different random players
        player_first = random.randint(1, num_players)
        player_second = random.randint(1, num_players)
        
        # Make sure the players are different
        while player_second == player_first:
            player_second = random.randint(1, num_players)
            
        # Randomly determine the result: 'w' for player_first win, 'l' for player_first loss
        result = random.choice(['w', 'l'])
        
        games.append((player_first, player_second, result))
    
    return games


def save_to_csv(games, output_file='games.csv'):
    """
    Save the game results to a CSV file.
    
    Args:
        games (list): List of tuples containing (player_first, player_second, result)
        output_file (str): Name of the output CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['game_no', 'player_first', 'player_second', 'result'])
        
        # Write game results
        for i, game in enumerate(games, 1):
            writer.writerow([i, *game])
            
    print(f"Successfully saved {len(games)} game results to {output_file}")


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Generate random chess game results.')
    parser.add_argument('-p', '--players', type=int, required=True, 
                        help='Number of players')
    parser.add_argument('-g', '--games', type=int, required=True,
                        help='Number of games to generate')
    parser.add_argument('-o', '--output', type=str, default='games.csv',
                        help='Output CSV file name (default: games.csv)')
    
    args = parser.parse_args()
    
    # Validate input
    if args.players < 2:
        print("Error: At least 2 players are required.")
        return
    
    if args.games < 1:
        print("Error: At least 1 game must be generated.")
        return
    
    # Generate random games
    games = generate_random_games(args.players, args.games)
    
    # Save to CSV
    save_to_csv(games, args.output)


if __name__ == '__main__':
    main()
