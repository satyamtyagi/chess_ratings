#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter to use Dirty Graph algorithm within the single-pass framework.
This can be imported into bt_single_pass_baselines_opdn_v3_fixed.py
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

# Counter for expected score calculations
expected_score_calls = 0

@dataclass
class Edge:
    """Edge storing observed wins and cached expected counts for each endpoint."""
    player1: int
    player2: int
    wins1: int = 0
    wins2: int = 0
    contribution1: float = 0.0
    contribution2: float = 0.0
    dirty_for1: bool = True
    dirty_for2: bool = True
    
    def add_game_result(self, winner: int, loser: int):
        if winner == self.player1:
            self.wins1 += 1
        else:
            self.wins2 += 1
        # Mark as dirty for both players
        self.dirty_for1 = self.dirty_for2 = True
    
    def total_games(self):
        return self.wins1 + self.wins2

@dataclass
class Node:
    """Node for a player."""
    id: int
    rating: float = 0.0
    edges: Dict[int, Edge] = field(default_factory=dict)
    actual: int = 0
    expected: float = 0.0
    
    def add_edge(self, edge: Edge, opponent_id: int):
        self.edges[opponent_id] = edge

def compute_expected_score(r1: float, r2: float) -> float:
    """Compute σ(r1 - r2) with clamping for stability."""
    global expected_score_calls
    expected_score_calls += 1
    x = max(min(r1 - r2, 100.0), -100.0)  # Clamping to ±100
    return 1.0 / (1.0 + math.exp(-x))

def refresh_edge_expected(edge: Edge, node1: Node, node2: Node) -> None:
    """
    Keep node.expected equal to sum of edge expected counts.
    Remove old cached E from BOTH endpoints, recompute with current ratings, add back.
    node1 must be Node(edge.player1); node2 must be Node(edge.player2).
    """
    # Remove old expected scores
    node1.expected -= edge.contribution1
    node2.expected -= edge.contribution2
    
    # Compute new expected scores
    n = edge.total_games()
    if n == 0:
        edge.contribution1, edge.contribution2 = 0.0, 0.0
        return
        
    p1 = compute_expected_score(node1.rating, node2.rating)
    E1 = n * p1
    E2 = n - E1
    
    # Update contributions
    edge.contribution1, edge.contribution2 = E1, E2
    node1.expected += E1
    node2.expected += E2

def process_dirty_edges_for_player(nodes, player_idx, learning_rate=0.01):
    """Process dirty edges for a player. Return True if any were dirty."""
    node = nodes[player_idx]
    any_dirty = False
    
    # Process dirty edges
    for opp_idx, edge in node.edges.items():
        opp_node = nodes[opp_idx]
        
        # Check if edge is dirty for this player
        if ((edge.player1 == player_idx and edge.dirty_for1) or 
            (edge.player2 == player_idx and edge.dirty_for2)):
            any_dirty = True
            
            # Update expected scores
            if edge.player1 == player_idx:
                refresh_edge_expected(edge, node, opp_node)
                edge.dirty_for1 = False  # Mark as clean for this player
            else:
                refresh_edge_expected(edge, opp_node, node)
                edge.dirty_for2 = False  # Mark as clean for this player
    
    # Update player rating if any edges were dirty
    if any_dirty:
        gradient = node.actual - node.expected
        node.rating += learning_rate * gradient
        
        # Mark all edges to neighbors as dirty (for the neighbor's side)
        for opp_idx, edge in node.edges.items():
            if edge.player1 == player_idx:
                edge.dirty_for2 = True  # Mark as dirty for opponent
            else:
                edge.dirty_for1 = True  # Mark as dirty for opponent
    
    return any_dirty

def onepass_dirty_graph(matches: List[Tuple[str,str,bool]], players: List[str], 
                        learning_rate: float = 0.01, phase2_iters: int = 5) -> Dict[str,float]:
    """
    Dirty Graph algorithm for Bradley-Terry, adapted for single-pass framework.
    
    Args:
        matches: List of tuples (player1, player2, result), where result is True if player1 won
        players: List of player IDs
        learning_rate: Learning rate for rating updates
        phase2_iters: Max iterations for phase 2 cleaning
    
    Returns:
        Dictionary mapping player IDs to their final ratings
    """
    # Reset counter
    global expected_score_calls
    expected_score_calls = 0
    
    # Create player index mapping
    player_to_idx = {p: i for i, p in enumerate(players)}
    idx_to_player = {i: p for i, p in enumerate(players)}
    
    # Initialize graph
    nodes = {i: Node(i) for i in range(len(players))}
    
    print(f"Using adaptive learning rate: {learning_rate:.6f} (players/games = {len(players)}/{len(matches)})")
    
    # Phase 1: Stream games
    print("\nPhase 1 Ratings:")
    for p1, p2, w1 in matches:
        i, j = player_to_idx[p1], player_to_idx[p2]
        
        # Skip self-matches
        if i == j:
            continue
        
        # Create edge if it doesn't exist
        if j not in nodes[i].edges:
            edge = Edge(i, j)
            nodes[i].add_edge(edge, j)
            nodes[j].add_edge(edge, i)
        
        # Update edge with game result
        winner, loser = (i, j) if w1 else (j, i)
        edge = nodes[winner].edges[loser]
        
        if edge.player1 == winner:
            edge.wins1 += 1
            nodes[winner].actual += 1
        else:
            edge.wins2 += 1
            nodes[winner].actual += 1
        
        # Mark dirty
        edge.dirty_for1 = edge.dirty_for2 = True
        
        # Process dirty edges for both players
        process_dirty_edges_for_player(nodes, winner, learning_rate)
        process_dirty_edges_for_player(nodes, loser, learning_rate)
    
    # Output Phase 1 ratings if verbose
    # (Would need to be properly formatted for the framework)
    
    # Phase 2: Clean remaining dirty edges
    print("\nPhase 2: Cleaning remaining dirty edges...")
    for _ in range(phase2_iters):
        any_dirty = False
        for player_idx in range(len(players)):
            if process_dirty_edges_for_player(nodes, player_idx, learning_rate):
                any_dirty = True
        if not any_dirty:
            break
    
    # Extract final ratings and normalize
    ratings = {p: nodes[player_to_idx[p]].rating for p in players}
    mean_rating = sum(ratings.values()) / len(ratings)
    final_ratings = {p: r - mean_rating for p, r in ratings.items()}
    
    print(f"\nFinal Ratings:")
    print(f"Total expected score calculations: {expected_score_calls}")
    
    return final_ratings

# Add to algorithm_builders in main()
"""
In bt_single_pass_baselines_opdn_v3_fixed.py, add this to the algorithm_builders list:

("DirtyGraph", lambda: onepass_dirty_graph(matches, players, learning_rate=0.01, phase2_iters=5)),
"""
