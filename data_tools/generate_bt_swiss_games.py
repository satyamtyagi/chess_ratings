#!/usr/bin/env python3
"""
Generate Swiss tournament games using Bradley-Terry model.
Players have thetas loaded from CSV file and games are generated
using Swiss pairing with BT win probabilities: P(i beats j) = 1 / (1 + exp(theta_j - theta_i))
"""

import argparse
import csv
import random
import math
from collections import defaultdict


def read_player_thetas(filename: str) -> dict:
    """Read player thetas from CSV file."""
    thetas = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = str(row['player_id'])
            theta = float(row['theta'])
            thetas[player_id] = theta
    return thetas


def bt_win_probability(theta1: float, theta2: float) -> float:
    """
    Calculate probability that player 1 beats player 2 using Bradley-Terry formula.
    
    Args:
        theta1: Player 1's theta (log-strength)
        theta2: Player 2's theta (log-strength)
        
    Returns:
        float: Probability that player 1 wins (0.0 to 1.0)
    """
    # Bradley-Terry: P(Player 1 wins) = 1 / (1 + exp(theta2 - theta1))
    return 1.0 / (1.0 + math.exp(theta2 - theta1))


def pair_players_swiss(scores: dict, played_pairs: set, players: list) -> list:
    """
    Pair players according to Swiss tournament rules.
    
    Args:
        scores (dict): Dictionary mapping player_id to current score
        played_pairs (set): Set of tuples representing pairs that have already played
        players (list): List of all player IDs
        
    Returns:
        list: List of pairs (player1, player2)
    """
    # Group players by score
    players_by_score = defaultdict(list)
    for player in players:
        players_by_score[scores[player]].append(player)
    
    # Sort score groups in descending order
    sorted_scores = sorted(players_by_score.keys(), reverse=True)
    
    unpaired = []
    for score in sorted_scores:
        unpaired.extend(players_by_score[score])
    
    pairs = []
    paired = set()
    
    # Try to pair players with similar scores
    while len(unpaired) >= 2:
        player1 = unpaired[0]
        unpaired.remove(player1)
        paired.add(player1)
        
        # Find best opponent for player1
        best_opponent = None
        for i, player2 in enumerate(unpaired):
            pair_key = tuple(sorted([player1, player2]))
            if pair_key not in played_pairs:
                best_opponent = player2
                break
        
        if best_opponent is None:
            # No valid opponent found, try next player
            if unpaired:
                best_opponent = unpaired[0]
                print(f"Warning: Forced pairing {player1} vs {best_opponent} (already played)")
            else:
                print(f"Warning: Player {player1} gets a bye")
                break
        
        unpaired.remove(best_opponent)
        paired.add(best_opponent)
        pairs.append((player1, best_opponent))
        played_pairs.add(tuple(sorted([player1, best_opponent])))
    
    # Handle bye if odd number of players
    if unpaired:
        print(f"Player {unpaired[0]} gets a bye this round")
    
    return pairs


def generate_swiss_tournament(player_thetas: dict, num_rounds: int) -> list:
    """
    Generate a Swiss tournament with Bradley-Terry probabilities.
    
    Args:
        player_thetas (dict): Dictionary mapping player IDs to their theta values
        num_rounds (int): Number of rounds to play
        
    Returns:
        list: List of game tuples (round, game_no, player1, player2, result)
    """
    players = list(player_thetas.keys())
    num_players = len(players)
    
    # Initialize scores
    scores = {player: 0.0 for player in players}
    played_pairs = set()
    games = []
    game_counter = 1
    
    print(f"Starting Swiss tournament: {num_players} players, {num_rounds} rounds")
    
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}:")
        
        # Pair players
        pairs = pair_players_swiss(scores, played_pairs, players)
        
        # Play games
        round_games = []
        for player1, player2 in pairs:
            # Get thetas
            theta1 = player_thetas[player1]
            theta2 = player_thetas[player2]
            
            # Calculate win probability for player1
            win_prob = bt_win_probability(theta1, theta2)
            
            # Determine result
            if random.random() < win_prob:
                result = 'w'  # Player 1 wins
                scores[player1] += 1.0
                winner, loser = player1, player2
            else:
                result = 'l'  # Player 1 loses (Player 2 wins)
                scores[player2] += 1.0
                winner, loser = player2, player1
            
            game = (round_num, game_counter, player1, player2, result)
            games.append(game)
            round_games.append((player1, player2, winner))
            game_counter += 1
        
        # Print round results
        for p1, p2, winner in round_games:
            theta1, theta2 = player_thetas[p1], player_thetas[p2]
            prob = bt_win_probability(theta1, theta2)
            print(f"  {p1} (θ={theta1:.3f}) vs {p2} (θ={theta2:.3f}) -> {winner} wins (p={prob:.3f})")
        
        # Print standings
        standings = sorted(players, key=lambda p: scores[p], reverse=True)
        print(f"  Standings: {', '.join([f'{p}({scores[p]})' for p in standings[:5]])}...")
    
    return games


def save_games_to_csv(games: list, filename: str):
    """Save games to CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'game_no', 'player_first', 'player_second', 'result'])
        for game in games:
            writer.writerow(game)
    print(f"\nSwiss tournament games saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate Swiss tournament with Bradley-Terry probabilities')
    parser.add_argument('--thetas', type=str, required=True,
                       help='CSV file containing player thetas')
    parser.add_argument('--rounds', type=int, default=7,
                       help='Number of rounds to play (default: 7)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='datasets/swiss_bt_games.csv',
                       help='Output filename for games CSV (default: datasets/swiss_bt_games.csv)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Read player thetas
    print(f"Reading player thetas from {args.thetas}")
    player_thetas = read_player_thetas(args.thetas)
    
    print(f"Loaded {len(player_thetas)} players:")
    theta_values = list(player_thetas.values())
    print(f"  Theta range: {min(theta_values):.3f} to {max(theta_values):.3f}")
    print(f"  Theta mean: {sum(theta_values)/len(theta_values):.3f}")
    
    # Generate Swiss tournament
    games = generate_swiss_tournament(player_thetas, args.rounds)
    
    # Save results
    save_games_to_csv(games, args.output)
    
    # Statistics
    total_games = len(games)
    wins_first = sum(1 for game in games if game[4] == 'w')
    losses_first = sum(1 for game in games if game[4] == 'l')
    
    print(f"\nTournament completed:")
    print(f"  Total games: {total_games}")
    print(f"  Results: {wins_first} wins, {losses_first} losses for player_first")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
