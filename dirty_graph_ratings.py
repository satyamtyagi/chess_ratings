#!/usr/bin/env python3
"""
Dirty Graph Ratings Algorithm

This script implements a graph-based rating algorithm where:
- Players are nodes with ratings
- Games are edges with contribution values
- Ratings are updated incrementally based on dirty edges
- Final ratings are converted to ELO scale
"""

import argparse
import csv
import math
import numpy as np
from collections import defaultdict


class Edge:
    """An edge represents games between two players."""
    def __init__(self, player1, player2):
        self.player1 = min(player1, player2)  # Ensure consistent ordering
        self.player2 = max(player1, player2)
        self.wins1 = 0  # Number of wins for player1
        self.wins2 = 0  # Number of wins for player2
        # Store separate contributions for each player
        self.contribution1 = 0  # Contribution to player1's rating
        self.contribution2 = 0  # Contribution to player2's rating
        # Track dirty status separately for each player
        self.dirty_for1 = True  # Whether this edge is dirty for player1
        self.dirty_for2 = True  # Whether this edge is dirty for player2
    
    def __repr__(self):
        return f"Edge({self.player1}-{self.player2}, W1:{self.wins1}, W2:{self.wins2}, C1:{self.contribution1:.4f}, C2:{self.contribution2:.4f}, D1:{self.dirty_for1}, D2:{self.dirty_for2})"
    
    def add_game_result(self, winner, loser):
        """Add game result to edge."""
        if winner == self.player1 and loser == self.player2:
            self.wins1 += 1
        elif winner == self.player2 and loser == self.player1:
            self.wins2 += 1
        else:
            raise ValueError(f"Players {winner}, {loser} don't match edge {self.player1}-{self.player2}")
        # Mark edge as dirty for both players after new game
        self.dirty_for1 = True
        self.dirty_for2 = True


class Node:
    """A node represents a player with rating and edges."""
    def __init__(self, player_id):
        self.id = player_id
        self.rating = 0  # Initial rating is 0
        self.edges = {}  # Dictionary of edges keyed by opponent ID
    
    def __repr__(self):
        return f"Node({self.id}, R:{self.rating:.4f}, E:{len(self.edges)})"
    
    def add_edge(self, edge, opponent_id):
        """Add or update edge for this node."""
        self.edges[opponent_id] = edge
    
    def get_edge(self, opponent_id):
        """Get edge connecting to opponent."""
        return self.edges.get(opponent_id, None)


def load_games_from_csv(file_path):
    """
    Load game data from a CSV file.
    
    Args:
        file_path (str): Path to CSV file containing game data
        
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
                'result': row['result']  # 'w' if player_first won, 'l' if player_second won
            }
            games.append(game)
    
    return games


def get_player_list(games):
    """
    Extract unique player IDs from game data.
    
    Args:
        games (list): List of dictionaries containing game data
        
    Returns:
        list: List of unique player IDs
    """
    players = set()
    for game in games:
        players.add(game['player_first'])
        players.add(game['player_second'])
    return sorted(list(players))


def initialize_graph(players):
    """
    Initialize graph with nodes for each player.
    
    Args:
        players (list): List of unique player IDs
    
    Returns:
        dict: Dictionary of Node objects keyed by player ID
    """
    nodes = {}
    for player in players:
        nodes[player] = Node(player)
    return nodes


def compute_expected_score(rating1, rating2):
    """
    Compute expected score for a player using Bradley-Terry model.
    
    Args:
        rating1 (float): Rating of the first player
        rating2 (float): Rating of the second player
    
    Returns:
        float: Expected score (probability of winning) for the first player
    """
    # Using the standard Bradley-Terry formula (same as in bradley_terry.py)
    return 1.0 / (1.0 + math.exp(rating2 - rating1))


def compute_edge_contributions(edge, node1, node2, learning_rate=0.1):
    """
    Compute edge contributions for both players based on game results and current ratings.
    
    Args:
        edge (Edge): Edge between two players
        node1 (Node): First player node
        node2 (Node): Second player node
        learning_rate (float): Learning rate for contribution calculation
    
    Returns:
        tuple: (contribution1, contribution2) - contributions to each player's rating
    """
    # Skip contribution calculation for edges with no games
    if edge.wins1 + edge.wins2 == 0:
        return 0, 0
    
    # Calculate expected scores based on current ratings
    expected1 = compute_expected_score(node1.rating, node2.rating)
    expected2 = compute_expected_score(node2.rating, node1.rating)
    
    # Calculate actual win ratios
    total_games = edge.wins1 + edge.wins2
    actual1 = edge.wins1 / total_games if total_games > 0 else 0
    actual2 = edge.wins2 / total_games if total_games > 0 else 0
    
    # Calculate contributions for each player
    contribution1 = learning_rate * (actual1 - expected1)
    contribution2 = learning_rate * (actual2 - expected2)
    
    return contribution1, contribution2


def process_games(games, learning_rate=1.0):
    """
    Process games and compute ratings using dirty graph approach.
    
    Args:
        games (list): List of dictionaries containing game data
        learning_rate (float): Learning rate for rating updates (default: 1.0)
    
    Returns:
        tuple: (ratings_dict, wins_dict) - Dictionary of final player ratings and dictionary of win counts
    """
    # Get unique players
    players = get_player_list(games)
    
    # Initialize graph with nodes for each player
    nodes = initialize_graph(players)
    
    # Create edges and process games
    edges = {}  # Dictionary to store all edges
    
    # Initialize win counts for each player
    win_counts = {player: 0 for player in players}
    
    # Process all games to build the graph structure
    for game in games:
        player1 = game['player_first']
        player2 = game['player_second']
        
        # Create edge key (always sorted for consistency)
        edge_key = (min(player1, player2), max(player1, player2))
        
        # Get or create edge
        if edge_key not in edges:
            edge = Edge(player1, player2)
            edges[edge_key] = edge
            
            # Add edge to both nodes
            nodes[player1].add_edge(edge, player2)
            nodes[player2].add_edge(edge, player1)
        else:
            edge = edges[edge_key]
        
        # Add game result to edge and update win counts
        if game['result'] == 'w':
            edge.add_game_result(player1, player2)
            win_counts[player1] += 1
        else:  # 'l'
            edge.add_game_result(player2, player1)
            win_counts[player2] += 1
    
    # Initialize list of players with dirty edges
    players_with_dirty_edges = set(players)  # Start with all players
    
    # Process until no more dirty edges
    iteration = 0
    max_iterations = 100  # Safety to prevent infinite loops
    
    while players_with_dirty_edges and iteration < max_iterations:
        iteration += 1
        
        # Get one player with dirty edges
        player = players_with_dirty_edges.pop()
        node = nodes[player]
        
        # Track if player's rating changes
        old_rating = node.rating
        
        # Process all dirty edges for this player
        rating_change = False
        
        for opponent_id, edge in node.edges.items():
            # Check if edge is dirty for this player
            is_dirty = (edge.player1 == player and edge.dirty_for1) or \
                      (edge.player2 == player and edge.dirty_for2)
            
            if is_dirty:
                # Get opponent node
                opponent_node = nodes[opponent_id]
                
                # Store old contributions
                if edge.player1 == player:
                    old_contribution = edge.contribution1
                else:
                    old_contribution = edge.contribution2
                
                # Calculate new contributions
                if edge.player1 == player:
                    node1, node2 = node, opponent_node
                else:
                    node1, node2 = opponent_node, node
                
                # Get new contributions
                contrib1, contrib2 = compute_edge_contributions(edge, node1, node2, learning_rate)
                
                # Update contributions in the edge
                if edge.player1 == player:
                    edge.contribution1 = contrib1
                    new_contribution = contrib1
                    edge.dirty_for1 = False  # Mark clean for this player
                else:
                    edge.contribution2 = contrib2
                    new_contribution = contrib2
                    edge.dirty_for2 = False  # Mark clean for this player
                
                # Update player's rating
                node.rating = node.rating - old_contribution + new_contribution
                rating_change = True
    
                # If player's rating changed, mark all edges of opponents as dirty
                if rating_change:
                    # Add opponents back to the processing queue
                    players_with_dirty_edges.add(opponent_id)
                    
                    # Mark all edges of opponent as dirty (for the opponent)
                    for opp_opp_id, opp_edge in opponent_node.edges.items():
                        if opp_edge.player1 == opponent_id:
                            opp_edge.dirty_for1 = True
                        else:
                            opp_edge.dirty_for2 = True
        
        # If no changes were made to this player's rating, no need to recheck
        if not rating_change:
            continue
    
    # Normalize ratings to have zero mean
    rating_sum = sum(node.rating for node in nodes.values())
    rating_mean = rating_sum / len(nodes)
    
    for node in nodes.values():
        node.rating -= rating_mean
    
    # Return dictionary mapping player IDs to ratings and dictionary of win counts
    ratings = {player: nodes[player].rating for player in players}
    return ratings, win_counts


def convert_to_elo(bt_rating):
    """
    Convert Bradley-Terry rating to ELO scale.
    
    Args:
        bt_rating (float): Bradley-Terry rating
    
    Returns:
        float: ELO rating
    """
    return 1200 + (400 / math.log(10)) * bt_rating


def save_ratings_to_csv(ratings, wins, file_path):
    """
    Save ratings and win statistics to a CSV file.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their ratings
        wins (dict): Dictionary mapping player IDs to their win counts
        file_path (str): Path to save ratings CSV
    """
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['player_no', 'bt_rating', 'elo_rating', 'wins'])
        
        for player, bt_rating in ratings.items():
            elo_rating = convert_to_elo(bt_rating)
            win_count = wins.get(player, 0)
            writer.writerow([player, f"{bt_rating:.6f}", f"{elo_rating:.1f}", win_count])
    
    print(f"Successfully saved ratings to {file_path}")


def main():
    parser = argparse.ArgumentParser(description='Dirty Graph rating calculation for chess games.')
    parser.add_argument('-f', '--file', type=str, default='games.csv',
                        help='CSV file containing game data (default: games.csv)')
    parser.add_argument('-l', '--learning-rate', type=float, default=1.0,
                        help='Learning rate for rating updates (default: 1.0)')
    parser.add_argument('-o', '--output-csv', type=str, default='dirty_graph_ratings.csv',
                        help='Output CSV file for ratings (default: dirty_graph_ratings.csv)')
    
    args = parser.parse_args()
    
    print(f"Loading games from {args.file}...")
    games = load_games_from_csv(args.file)
    print(f"Loaded {len(games)} games involving {len(get_player_list(games))} players")
    
    print(f"\nRunning Dirty Graph Ratings algorithm...")
    print(f"Learning rate: {args.learning_rate}")
    
    ratings, win_counts = process_games(games, learning_rate=args.learning_rate)
    
    # Save ratings to CSV
    save_ratings_to_csv(ratings, win_counts, args.output_csv)
    
    # Print final ratings
    print("\nFinal Ratings:")
    print("-" * 60)
    print(f"{'Rank':4} | {'Player':6} | {'BT Rating':12} | {'ELO Rating':10} | {'Wins':5}")
    print("-" * 60)
    
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (player, bt_rating) in enumerate(sorted_ratings, 1):
        elo_rating = convert_to_elo(bt_rating)
        wins = win_counts[player]
        print(f"{rank:4} | {player:6} | {bt_rating:12.6f} | {elo_rating:10.1f} | {wins:5d}")
    
    # Print rating statistics
    bt_values = list(ratings.values())
    print("\nRating Statistics:")
    print(f"Mean Rating: {np.mean(bt_values):.6f}")
    print(f"Median Rating: {np.median(bt_values):.6f}")
    print(f"Min Rating: {min(bt_values):.6f}")
    print(f"Max Rating: {max(bt_values):.6f}")
    print(f"Standard Deviation: {np.std(bt_values):.6f}")


if __name__ == '__main__':
    main()
