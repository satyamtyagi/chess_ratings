#!/usr/bin/env python3
"""
Generate chess game results using Swiss tournament pairing rules.

This script creates a simulated Swiss tournament with a specified number of players.
The number of rounds is calculated as ceiling(log2(number_of_players)).
Players are paired according to Swiss tournament rules:
- Players are paired with others having the same score when possible
- No player plays against the same opponent twice
- Results are randomly determined

Output is saved to a CSV file with columns for game_no, player_first, player_second, and result (w/l).
"""

import argparse
import csv
import random
import math
from collections import defaultdict


def determine_rounds(num_players):
    """
    Calculate the number of rounds for a Swiss tournament.
    
    Args:
        num_players (int): Number of players
        
    Returns:
        int: Number of rounds (ceiling of log base 2 of number of players)
    """
    return math.ceil(math.log2(num_players))


def pair_players(players_by_score, played_pairs, num_players):
    """
    Pair players according to Swiss tournament rules.
    
    Args:
        players_by_score (dict): Dictionary mapping scores to lists of players
        played_pairs (set): Set of tuples representing pairs that have already played
        num_players (int): Total number of players
        
    Returns:
        list: List of pairs (player1, player2)
    """
    # Create a list of all players
    all_players = list(range(1, num_players + 1))
    
    # Sort scores in descending order
    sorted_scores = sorted(players_by_score.keys(), reverse=True)
    
    # Keep track of players who have been paired
    paired_players = set()
    pairs = []
    
    # First, try to pair players with the same score
    for score in sorted_scores:
        players = players_by_score[score].copy()  # Make a copy to avoid modifying during iteration
        # Shuffle players with the same score to ensure random pairings
        random.shuffle(players)
        
        # Try to pair players with the same score
        while len(players) >= 2:
            player1 = players.pop(0)
            if player1 in paired_players:
                continue
                
            # Find a valid opponent with the same score
            found_opponent = False
            for i, player2 in enumerate(players):
                if player2 not in paired_players and (player1, player2) not in played_pairs and (player2, player1) not in played_pairs:
                    pairs.append((player1, player2))
                    paired_players.add(player1)
                    paired_players.add(player2)
                    players.pop(i)
                    found_opponent = True
                    break
            
            # If no valid opponent found in the same score group, add back to the player pool
            if not found_opponent:
                continue  # We'll handle this player in the next phase
    
    # If there are unpaired players, pair them across score groups
    unpaired = [p for p in all_players if p not in paired_players]
    
    # Create a compatibility graph for remaining players
    # For each player, find all possible opponents they haven't played yet
    possible_opponents = {}
    for p1 in unpaired:
        possible_opponents[p1] = [p2 for p2 in unpaired if p1 != p2 and 
                               (p1, p2) not in played_pairs and 
                               (p2, p1) not in played_pairs]
    
    # While there are still players to pair
    while len(unpaired) >= 2:
        # Find player with fewest remaining possible opponents
        player1 = min(unpaired, key=lambda p: len(possible_opponents[p]) if possible_opponents[p] else float('inf'))
        
        # If this player has no possible opponents, we need to break a constraint
        if not possible_opponents[player1]:
            # In a real tournament, we'd need more complex logic here
            # For our simulation, we'll just pair with someone they've already played
            possible = [p for p in unpaired if p != player1]
            if not possible:
                break  # No opponents left at all
            player2 = random.choice(possible)
        else:
            # Otherwise, pair with a valid opponent
            player2 = random.choice(possible_opponents[player1])
        
        # Add the pair
        pairs.append((player1, player2))
        
        # Remove both players from the unpaired list
        unpaired.remove(player1)
        unpaired.remove(player2)
        
        # Update the possible opponents for all remaining players
        for p in unpaired:
            if player1 in possible_opponents[p]:
                possible_opponents[p].remove(player1)
            if player2 in possible_opponents[p]:
                possible_opponents[p].remove(player2)
                
    # If there's a single unpaired player (odd number), they get a bye
    # In a real tournament, no player should get more than one bye
    
    return pairs


def generate_swiss_games(num_players):
    """
    Generate game results for a Swiss-style tournament.
    
    Args:
        num_players (int): Number of players
        
    Returns:
        list: List of tuples containing (game_no, player_first, player_second, result)
    """
    num_rounds = determine_rounds(num_players)
    print(f"Swiss tournament with {num_players} players will have {num_rounds} rounds")
    
    # Keep track of player scores and pairings
    scores = defaultdict(int)  # Player -> Score
    played_pairs = set()  # Set of (player1, player2) tuples
    games = []
    game_no = 1
    
    for round_num in range(1, num_rounds + 1):
        print(f"Generating round {round_num}...")
        
        # Group players by score
        players_by_score = defaultdict(list)
        for player in range(1, num_players + 1):
            players_by_score[scores[player]].append(player)
            
        # Pair players for this round
        pairs = pair_players(players_by_score, played_pairs, num_players)
        
        # Generate results for each pair and update scores
        for player1, player2 in pairs:
            # Randomly determine the result
            result = random.choice(['w', 'l'])
            
            # Record the game
            games.append((game_no, player1, player2, result))
            game_no += 1
            
            # Update scores
            if result == 'w':
                scores[player1] += 1
            else:
                scores[player2] += 1
                
            # Mark that these players have played each other
            played_pairs.add((player1, player2))
    
    return games


def save_to_csv(games, output_file='swiss_games.csv'):
    """
    Save the game results to a CSV file.
    
    Args:
        games (list): List of tuples containing (game_no, player_first, player_second, result)
        output_file (str): Name of the output CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['game_no', 'player_first', 'player_second', 'result'])
        
        # Write game results
        for game in games:
            writer.writerow(game)
            
    print(f"Successfully saved {len(games)} Swiss tournament game results to {output_file}")


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Generate Swiss tournament chess game results.')
    parser.add_argument('-p', '--players', type=int, required=True, 
                        help='Number of players')
    parser.add_argument('-o', '--output', type=str, default='swiss_games.csv',
                        help='Output CSV file name (default: swiss_games.csv)')
    
    args = parser.parse_args()
    
    # Validate input
    if args.players < 2:
        print("Error: At least 2 players are required.")
        return
    
    # Generate Swiss tournament games
    games = generate_swiss_games(args.players)
    
    # Save to CSV
    save_to_csv(games, args.output)


if __name__ == '__main__':
    main()
