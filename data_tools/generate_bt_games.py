#!/usr/bin/env python3
"""
Generate synthetic games using Bradley-Terry model directly.
Players have thetas drawn from Normal(0, sigma^2) and games are generated
using BT win probabilities: P(i beats j) = 1 / (1 + exp(theta_j - theta_i))
"""

import argparse
import csv
import random
import math
import numpy as np


def gen_theta(n_players: int, sigma: float = 1.0, seed: int = 42,
              labels=None, as_dict: bool = True):
    """
    Draws theta_i ~ Normal(0, sigma^2) for i=1..n_players.

    Args:
        n_players: number of players
        sigma: standard deviation (variance = sigma**2)
        seed: RNG seed for reproducibility
        labels: optional iterable of player ids; defaults to ["1", ..., "n"]
        as_dict: if True returns {player_id: theta}; else returns numpy array

    Returns:
        dict[str,float] or np.ndarray
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(loc=0.0, scale=sigma, size=n_players).astype(float)
    if not as_dict:
        return theta
    if labels is None:
        labels = [str(i+1) for i in range(n_players)]
    return dict(zip(labels, theta))


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


def generate_bt_games(num_players: int, num_games: int, player_thetas: dict) -> list:
    """
    Generate synthetic games using Bradley-Terry probabilities.
    
    Args:
        num_players (int): Number of players
        num_games (int): Number of games to generate
        player_thetas (dict): Dictionary mapping player IDs to their theta values
        
    Returns:
        list: List of game tuples (game_no, player1, player2, result)
    """
    games = []
    player_ids = list(player_thetas.keys())
    
    for game_no in range(1, num_games + 1):
        # Randomly select two different players
        player1 = random.choice(player_ids)
        player2 = random.choice(player_ids)
        while player2 == player1:
            player2 = random.choice(player_ids)
        
        # Get their thetas
        theta1 = player_thetas[player1]
        theta2 = player_thetas[player2]
        
        # Calculate win probability for player 1
        win_prob = bt_win_probability(theta1, theta2)
        
        # Determine result based on probability
        if random.random() < win_prob:
            result = 'w'  # Player 1 wins
        else:
            result = 'l'  # Player 1 loses (Player 2 wins)
        
        games.append((game_no, player1, player2, result))
    
    return games


def save_games_to_csv(games: list, filename: str):
    """Save games to CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['game_no', 'player_first', 'player_second', 'result'])
        for game in games:
            writer.writerow(game)
    print(f"Games saved to {filename}")


def save_thetas_to_csv(player_thetas: dict, filename: str):
    """Save player thetas to CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['player_id', 'theta'])
        for player_id in sorted(player_thetas.keys(), key=lambda x: int(x) if x.isdigit() else x):
            writer.writerow([player_id, player_thetas[player_id]])
    print(f"Player thetas saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic Bradley-Terry games')
    parser.add_argument('-p', '--players', type=int, default=20,
                       help='Number of players (default: 20)')
    parser.add_argument('-g', '--games', type=int, default=2000,
                       help='Number of games to generate (default: 2000)')
    parser.add_argument('--sigma', type=float, default=1.0,
                       help='Standard deviation for theta generation (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-games', type=str, default='datasets/bt_games.csv',
                       help='Output filename for games CSV (default: datasets/bt_games.csv)')
    parser.add_argument('--output-thetas', type=str, default='datasets/player_thetas.csv',
                       help='Output filename for player thetas CSV (default: datasets/player_thetas.csv)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Generating {args.players} players with Bradley-Terry thetas...")
    
    # Generate player thetas using the provided function
    player_thetas = gen_theta(n_players=args.players, sigma=args.sigma, seed=args.seed)
    
    # Display theta statistics
    theta_values = list(player_thetas.values())
    print(f"Player thetas: min={min(theta_values):.3f}, max={max(theta_values):.3f}, mean={sum(theta_values)/len(theta_values):.3f}")
    print(f"Standard deviation: {np.std(theta_values):.3f}")
    
    print(f"Generating {args.games} games with Bradley-Terry probabilities...")
    
    # Generate games
    games = generate_bt_games(args.players, args.games, player_thetas)
    
    # Save results
    save_games_to_csv(games, args.output_games)
    save_thetas_to_csv(player_thetas, args.output_thetas)
    
    # Calculate win/loss statistics
    wins_first = sum(1 for _, _, _, result in games if result == 'w')
    losses_first = sum(1 for _, _, _, result in games if result == 'l')
    
    print(f"\nGenerated files:")
    print(f"  Games: {args.output_games}")
    print(f"  Thetas: {args.output_thetas}")
    print(f"Results: {wins_first} wins, {losses_first} losses for player_first")


if __name__ == '__main__':
    main()
