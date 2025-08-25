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
    
    # Calculate total games/wins
    total_wins = edge.wins1 + edge.wins2
    
    # Calculate contributions for each player using raw win counts and expected wins
    contribution1 = learning_rate * (edge.wins1 - total_wins * expected1)
    contribution2 = learning_rate * (edge.wins2 - total_wins * expected2)
    
    return contribution1, contribution2


def process_games(games, learning_rate=1.0):
    """
    Process games and compute ratings using dirty graph approach with two phases:
    1. Process each game individually, updating ratings and marking other edges as dirty
    2. Clean remaining dirty edges after all games are processed
    
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
    
    # Dictionary to store all edges
    edges = {}
    
    # Initialize win counts for each player
    win_counts = {player: 0 for player in players}
    
    # Phase 1: Process each game individually
    print("\nPhase 1: Processing individual games...")
    game_count = 0
    
    for game in games:
        game_count += 1
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
        
        # Update win counts based on game result
        if game['result'] == 'w':
            edge.add_game_result(player1, player2)
            win_counts[player1] += 1
            winner, loser = player1, player2
        else:  # 'l'
            edge.add_game_result(player2, player1)
            win_counts[player2] += 1
            winner, loser = player2, player1
            
        # First process all dirty edges for both players involved in the game
        for player in [player1, player2]:
            player_node = nodes[player]
            
            # Process all dirty edges for this player
            for opponent_id, other_edge in player_node.edges.items():
                # Check if this edge is dirty for this player
                is_dirty = (other_edge.player1 == player and other_edge.dirty_for1) or \
                          (other_edge.player2 == player and other_edge.dirty_for2)
                
                if is_dirty:
                    # Get opponent node
                    opponent_node = nodes[opponent_id]
                    
                    # Store old contributions
                    if other_edge.player1 == player:
                        old_contribution = other_edge.contribution1
                    else:
                        old_contribution = other_edge.contribution2
                    
                    # Calculate new contributions
                    if other_edge.player1 == player:
                        node1, node2 = player_node, opponent_node
                    else:
                        node1, node2 = opponent_node, player_node
                    
                    # Get new contributions
                    contrib1, contrib2 = compute_edge_contributions(other_edge, node1, node2, learning_rate)
                    
                    # Update contributions in the edge
                    if other_edge.player1 == player:
                        other_edge.contribution1 = contrib1
                        new_contribution = contrib1
                        other_edge.dirty_for1 = False  # Mark clean for this player
                    else:
                        other_edge.contribution2 = contrib2
                        new_contribution = contrib2
                        other_edge.dirty_for2 = False  # Mark clean for this player
                    
                    # Update player's rating
                    player_node.rating = player_node.rating - old_contribution + new_contribution
        
        # Now process the current game
        winner_node = nodes[winner]
        loser_node = nodes[loser]
        
        # Calculate and store contributions for the specific edge
        if edge.player1 == winner:
            old_contrib1 = edge.contribution1
            old_contrib2 = edge.contribution2
            contrib1, contrib2 = compute_edge_contributions(edge, winner_node, loser_node, learning_rate)
            edge.contribution1 = contrib1
            edge.contribution2 = contrib2
            
            # Update ratings
            winner_node.rating = winner_node.rating - old_contrib1 + contrib1
            loser_node.rating = loser_node.rating - old_contrib2 + contrib2
        else:
            old_contrib1 = edge.contribution1
            old_contrib2 = edge.contribution2
            contrib1, contrib2 = compute_edge_contributions(edge, loser_node, winner_node, learning_rate)
            edge.contribution1 = contrib1
            edge.contribution2 = contrib2
            
            # Update ratings
            loser_node.rating = loser_node.rating - old_contrib1 + contrib1
            winner_node.rating = winner_node.rating - old_contrib2 + contrib2
        
        # Mark all edges of these two players as dirty (except the one just played)
        for player, player_node in [(winner, winner_node), (loser, loser_node)]:
            for opponent_id, other_edge in player_node.edges.items():
                if opponent_id != winner and opponent_id != loser:  # Skip the edge just played
                    if other_edge.player1 == player:
                        other_edge.dirty_for1 = True
                    else:
                        other_edge.dirty_for2 = True
        
        # For progress update, show ratings every 50 games
        if game_count % 50 == 0 or game_count == len(games):
            print(f"Processed {game_count}/{len(games)} games")
    
    # Normalize and show ratings after Phase 1
    normalize_ratings(nodes)
    phase1_ratings = {player: nodes[player].rating for player in players}
    print("\nPhase 1 Completed - Ratings after game processing:")
    print_ratings(phase1_ratings, win_counts)
    
    # Return Phase 1 ratings for saving to CSV
    phase1_result = phase1_ratings.copy()
    
    # Phase 2: Clean remaining dirty edges
    print("\nPhase 2: Cleaning remaining dirty edges...")
    
    # Count dirty edges for each player
    dirty_edge_counts = {}
    for player, node in nodes.items():
        dirty_edges = 0
        for opponent_id, edge in node.edges.items():
            if (edge.player1 == player and edge.dirty_for1) or \
               (edge.player2 == player and edge.dirty_for2):
                dirty_edges += 1
        dirty_edge_counts[player] = dirty_edges
    
    # Create set of players with dirty edges
    players_with_dirty_edges = {player for player, count in dirty_edge_counts.items() if count > 0}
    
    # Process until no more dirty edges
    iteration = 0
    max_iterations = 20  # Lower the max iterations since we expect fewer
    
    print(f"Starting Phase 2 with {len(players_with_dirty_edges)} players having dirty edges")
    total_dirty_edges = sum(dirty_edge_counts.values())
    print(f"Total dirty edges: {total_dirty_edges}")
    
    while players_with_dirty_edges and iteration < max_iterations:
        iteration += 1
        
        # Get one player with dirty edges
        player = players_with_dirty_edges.pop()
        node = nodes[player]
        player_dirty_edges_cleaned = 0
        
        # Print progress for each iteration
        print(f"Phase 2 iteration {iteration} - Processing player {player} with {dirty_edge_counts[player]} dirty edges")
        
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
                    dirty_edge_counts[player] -= 1
                    player_dirty_edges_cleaned += 1
                else:
                    edge.contribution2 = contrib2
                    new_contribution = contrib2
                    edge.dirty_for2 = False  # Mark clean for this player
                    dirty_edge_counts[player] -= 1
                    player_dirty_edges_cleaned += 1
                
                # Update player's rating but don't mark other edges as dirty
                # This is the key change - we don't want to keep creating new dirty edges
                node.rating = node.rating - old_contribution + new_contribution
        
        # Add player back to queue if they still have dirty edges
        if dirty_edge_counts[player] > 0:
            players_with_dirty_edges.add(player)
        
        print(f"  Cleaned {player_dirty_edges_cleaned} edges, {dirty_edge_counts[player]} dirty edges remaining")
        
        # Count total remaining dirty edges
        total_dirty_edges = sum(dirty_edge_counts.values())
        print(f"  Total remaining dirty edges across all players: {total_dirty_edges}")
        
        # If all edges are clean, we're done
        if total_dirty_edges == 0:
            break
    
    print(f"Phase 2 completed after {iteration} iterations")
    
    # Final normalization of ratings
    normalize_ratings(nodes)
    
    # Return dictionary mapping player IDs to ratings, phase 1 ratings, and dictionary of win counts
    phase2_ratings = {player: nodes[player].rating for player in players}
    return phase1_result, phase2_ratings, win_counts


def print_ratings(ratings, wins):
    """
    Print player ratings in sorted order.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to ratings
        wins (dict): Dictionary mapping player IDs to win counts
    """
    # Convert to ELO scale
    elo_ratings = {player: convert_to_elo(rating) for player, rating in ratings.items()}
    
    # Sort players by rating (descending)
    sorted_players = sorted(ratings.keys(), key=lambda x: ratings[x], reverse=True)
    
    # Calculate statistics
    bt_ratings = list(ratings.values())
    mean_rating = sum(bt_ratings) / len(bt_ratings)
    median_rating = sorted(bt_ratings)[len(bt_ratings) // 2]
    min_rating = min(bt_ratings)
    max_rating = max(bt_ratings)
    std_dev = math.sqrt(sum((r - mean_rating) ** 2 for r in bt_ratings) / len(bt_ratings))
    
    # Display sorted players and ratings
    print("------------------------------------------------------------")
    print("Rank | Player | BT Rating    | ELO Rating | Wins ")
    print("------------------------------------------------------------")
    
    for rank, player in enumerate(sorted_players, 1):
        print(f"{rank:4d} | {player:6d} | {ratings[player]:11.6f} | {elo_ratings[player]:9.1f} | {wins[player]:4d}")
    
    # Display statistics
    print("\nRating Statistics:")
    print(f"Mean Rating: {mean_rating:.6f}")
    print(f"Median Rating: {median_rating:.6f}")
    print(f"Min Rating: {min_rating:.6f}")
    print(f"Max Rating: {max_rating:.6f}")
    print(f"Standard Deviation: {std_dev:.6f}")



def normalize_ratings(nodes):
    """
    Normalize ratings to have zero mean.
    
    Args:
        nodes (dict): Dictionary of player nodes
    """
    rating_sum = sum(node.rating for node in nodes.values())
    rating_mean = rating_sum / len(nodes)
    
    for node in nodes.values():
        node.rating -= rating_mean


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
    parser.add_argument('--ph1-csv', type=str, default='ph1_dirty_graph_ratings.csv',
                        help='Output CSV file for Phase 1 ratings (default: ph1_dirty_graph_ratings.csv)')
    parser.add_argument('--ph2-csv', type=str, default='ph2_dirty_graph_ratings.csv',
                        help='Output CSV file for Phase 2 ratings (default: ph2_dirty_graph_ratings.csv)')
    
    args = parser.parse_args()
    
    print(f"Loading games from {args.file}...")
    games = load_games_from_csv(args.file)
    print(f"Loaded {len(games)} games involving {len(get_player_list(games))} players")
    
    print(f"\nRunning Dirty Graph Ratings algorithm...")
    print(f"Learning rate: {args.learning_rate}")
    
    # Process games using the two-phase approach
    phase1_ratings, phase2_ratings, win_counts = process_games(games, learning_rate=args.learning_rate)
    
    # Save Phase 1 ratings to CSV
    save_ratings_to_csv(phase1_ratings, win_counts, args.ph1_csv)
    
    # Save Phase 2 ratings to CSV
    save_ratings_to_csv(phase2_ratings, win_counts, args.ph2_csv)
    
    # For backwards compatibility, also save the final ratings to the original output file
    save_ratings_to_csv(phase2_ratings, win_counts, args.output_csv)
    
    # Print final ratings (now handled by print_ratings in process_games)
    print("\nFinal Ratings (after both phases):")
    print_ratings(phase2_ratings, win_counts)


if __name__ == '__main__':
    main()
