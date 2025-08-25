#!/usr/bin/env python3
"""
Dirty Graph Ratings (BT-gradient, dirty-edge, counts-weighted)

Minimal, surgical & production-ready version of the "v2" idea:
- Edge.contribution1/2 cache EXPECTED COUNTS (E1, E2), not gradients.
- Node has `actual` (wins count) and `expected` (sum of cached expectations).
- When touching a dirty edge, we "remove old / add new" expected counts for BOTH endpoints.
- **Important**: We do ONE gradient step per NODE after cleaning *all* its dirty edges.
- Two-phase flow:
  Phase 1 (stream): process each game, do one-hop dirty repairs & a per-node step;
                    then refresh the just-played edge and step winner/loser once more.
  Phase 2 (clean): clean any leftover dirty edges for each node (no neighbor fan-out), step once per node.
- Ratings are mean-centered; ELO = anchor + (400/ln10) * (BT_rating - anchor_bt).

CSV format expected: game_no, player_first, player_second, result
- result: 'w' means player_first won, anything else means player_second won.
"""

from __future__ import annotations
from typing import Dict, Tuple, List
from collections import defaultdict
from dataclasses import dataclass, field
import math
import csv
import argparse

# -------------------- Data structures --------------------

@dataclass
class Edge:
    """Edge storing observed wins and cached expected counts for each endpoint."""
    player1: int
    player2: int
    wins1: int = 0
    wins2: int = 0
    contribution1: float = 0.0  # expected wins credited to player1
    contribution2: float = 0.0  # expected wins credited to player2
    dirty_for1: bool = True
    dirty_for2: bool = True

    def __post_init__(self):
        # Normalize ordering so player1 < player2
        if self.player2 < self.player1:
            self.player1, self.player2 = self.player2, self.player1

    def add_game_result(self, winner: int, loser: int) -> None:
        if winner == self.player1 and loser == self.player2:
            self.wins1 += 1
        elif winner == self.player2 and loser == self.player1:
            self.wins2 += 1
        else:
            raise ValueError(f"Players {winner}, {loser} don't match edge {self.player1}-{self.player2}")
        self.dirty_for1 = True
        self.dirty_for2 = True

    def total_games(self) -> int:
        return self.wins1 + self.wins2


@dataclass
class Node:
    """Node for a player."""
    id: int
    rating: float = 0.0           # BT log-rating θ
    edges: Dict[int, Edge] = field(default_factory=dict)  # opp_id -> edge
    actual: int = 0               # total wins for this player (counts)
    expected: float = 0.0         # sum of expected wins from all incident edges

    def add_edge(self, edge: Edge, opponent_id: int) -> None:
        self.edges[opponent_id] = edge


# -------------------- IO helpers --------------------

def load_games_from_csv(file_path: str) -> List[dict]:
    """
    Load game data from a CSV with columns:
      - game_no (int), player_first (int), player_second (int), result ('w' if player_first won)
    """
    games: List[dict] = []
    with open(file_path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            game_no = int(row['game_no'])
            p1 = int(row['player_first'])
            p2 = int(row['player_second'])
            res = row['result'].strip().lower()
            if res == 'w':
                winner, loser = p1, p2
            else:
                winner, loser = p2, p1
            games.append({'game_no': game_no, 'p1': p1, 'p2': p2, 'winner': winner, 'loser': loser})
    games.sort(key=lambda r: r['game_no'])
    return games


def get_player_list(games: List[dict]) -> List[int]:
    players = set()
    for g in games:
        players.add(g['p1'])
        players.add(g['p2'])
    return sorted(players)


def initialize_graph(players: List[int]) -> Dict[int, Node]:
    return {p: Node(p) for p in players}


# -------------------- BT math helpers --------------------

def compute_expected_score(r1: float, r2: float) -> float:
    """σ(r1 - r2) with clamping for stability."""
    x = max(min(r1 - r2, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-x))


def compute_edge_expected(edge: Edge, node1: Node, node2: Node) -> Tuple[float, float]:
    """Return EXPECTED COUNTS (E1, E2) under CURRENT ratings. node1 is edge.player1, node2 is edge.player2."""
    n = edge.total_games()
    if n == 0:
        return 0.0, 0.0
    p1 = compute_expected_score(node1.rating, node2.rating)
    E1 = n * p1
    E2 = n - E1
    return E1, E2


def refresh_edge_expected(edge: Edge, node1: Node, node2: Node) -> None:
    """
    Keep node.expected equal to the sum of its edges' expected counts.
    Remove old cached E from BOTH endpoints, recompute with CURRENT ratings, add back, store on edge.
    node1 must be Node(edge.player1); node2 must be Node(edge.player2).
    """
    # Remove old
    node1.expected -= edge.contribution1
    node2.expected -= edge.contribution2

    # Recompute
    E1, E2 = compute_edge_expected(edge, node1, node2)

    # Store & add new
    edge.contribution1, edge.contribution2 = E1, E2
    node1.expected += E1
    node2.expected += E2


def normalize_ratings(nodes: Dict[int, Node]) -> None:
    if not nodes:
        return
    m = sum(n.rating for n in nodes.values()) / len(nodes)
    for n in nodes.values():
        n.rating -= m


def convert_to_elo(bt_rating: float, anchor_bt: float = 0.0, anchor_elo: float = 1200.0) -> float:
    """ELO = anchor_elo + (400/ln 10) * (bt_rating - anchor_bt)."""
    return anchor_elo + (400.0 / math.log(10.0)) * (bt_rating - anchor_bt)


def print_ratings(ratings: Dict[int, float], wins: Dict[int, int]) -> None:
    elo = {p: convert_to_elo(r) for p, r in ratings.items()}
    sorted_players = sorted(ratings, key=lambda x: ratings[x], reverse=True)
    print("Player |  BT (θ)    |  ELO    | Wins")
    print("-------------------------------------")
    for p in sorted_players:
        print(f"{p:6d} | {ratings[p]:9.4f} | {elo[p]:7.1f} | {wins.get(p,0):4d}")


# -------------------- Core algorithm --------------------

def train_with_dirty_edges(
    games: List[dict],
    learning_rate: float = 0.2,
    phase2_max_iters: int = 5,
    verbose_every: int = 0
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, int]]:
    """
    Two-phase processing:
      Phase 1: stream games; for each game,
               - repair all dirty edges for both endpoints (one hop)
               - do ONE node step per endpoint using (actual - expected)
               - refresh the just-played edge and step winner/loser once more
               - mark their other edges dirty on their own ends
      Phase 2: clean any leftover dirty edges per player (no neighbor fan-out), step once per player
    Returns:
      (phase1_ratings, final_ratings, win_counts)
    """
    players = get_player_list(games)
    nodes = initialize_graph(players)
    edges: Dict[Tuple[int,int], Edge] = {}
    wins = defaultdict(int)

    # -------- Phase 1 --------
    if verbose_every:
        print("\nPhase 1: Processing individual games...")

    for idx, g in enumerate(games, 1):
        p1, p2, winner, loser = g['p1'], g['p2'], g['winner'], g['loser']
        key = (min(p1,p2), max(p1,p2))

        # Ensure edge exists
        if key not in edges:
            e = Edge(*key)
            edges[key] = e
            # Attach to nodes
            nodes[e.player1].add_edge(e, e.player2)
            nodes[e.player2].add_edge(e, e.player1)
        else:
            e = edges[key]

        # Record the result
        e.add_game_result(winner, loser)
        nodes[winner].actual += 1
        wins[winner] += 1

        # Clean dirty edges for both endpoints and step ONCE per endpoint
        for player in (p1, p2):
            node = nodes[player]
            for opp, edge in list(node.edges.items()):
                is_dirty_for_player = (edge.player1 == player and edge.dirty_for1) or \
                                      (edge.player2 == player and edge.dirty_for2)
                if not is_dirty_for_player:
                    continue
                n1 = nodes[edge.player1]
                n2 = nodes[edge.player2]
                refresh_edge_expected(edge, n1, n2)
                if edge.player1 == player:
                    edge.dirty_for1 = False
                else:
                    edge.dirty_for2 = False
            # ONE gradient step per node after cleaning its incident dirty edges
            g_node = node.actual - node.expected
            node.rating += learning_rate * g_node

        # Refresh the current game edge based on the updated ratings, then step winner & loser once
        refresh_edge_expected(e, nodes[e.player1], nodes[e.player2])
        gw = nodes[winner].actual - nodes[winner].expected
        gl = nodes[loser].actual  - nodes[loser].expected
        nodes[winner].rating += learning_rate * gw
        nodes[loser].rating  += learning_rate * gl

        # Mark all OTHER edges of winner and loser dirty on their own ends
        for player in (winner, loser):
            for opp, edge in nodes[player].edges.items():
                if (edge.player1 == winner and edge.player2 == loser) or \
                   (edge.player1 == loser  and edge.player2 == winner):
                    continue
                if edge.player1 == player:
                    edge.dirty_for1 = True
                else:
                    edge.dirty_for2 = True

        if verbose_every and (idx % verbose_every == 0 or idx == len(games)):
            print(f"Processed {idx}/{len(games)} games")

    # Normalize & snapshot Phase 1
    normalize_ratings(nodes)
    phase1_ratings = {p: nodes[p].rating for p in players}

    # -------- Phase 2: clean remaining dirty edges (no neighbor fan-out) --------
    if verbose_every:
        print("\nPhase 2: Cleaning remaining dirty edges...")

    # Count dirty ends per player
    dirty_counts: Dict[int, int] = {}
    for p, node in nodes.items():
        cnt = 0
        for opp, edge in node.edges.items():
            if (edge.player1 == p and edge.dirty_for1) or (edge.player2 == p and edge.dirty_for2):
                cnt += 1
        dirty_counts[p] = cnt

    players_with_dirty = {p for p, c in dirty_counts.items() if c > 0}

    iters = 0
    while players_with_dirty and iters < phase2_max_iters:
        iters += 1
        p = players_with_dirty.pop()
        node = nodes[p]

        # Clean all dirty edges for this player
        for opp, edge in node.edges.items():
            is_dirty_for_p = (edge.player1 == p and edge.dirty_for1) or \
                             (edge.player2 == p and edge.dirty_for2)
            if not is_dirty_for_p:
                continue
            refresh_edge_expected(edge, nodes[edge.player1], nodes[edge.player2])
            if edge.player1 == p:
                edge.dirty_for1 = False
            else:
                edge.dirty_for2 = False
            dirty_counts[p] -= 1

        # ONE gradient step after cleaning
        g_node = node.actual - node.expected
        node.rating += learning_rate * g_node

        if dirty_counts[p] > 0:
            players_with_dirty.add(p)

    # Final normalize & return
    normalize_ratings(nodes)
    final_ratings = {p: nodes[p].rating for p in players}
    return phase1_ratings, final_ratings, dict(wins)


# -------------------- CLI --------------------

def save_ratings_to_csv(ratings: Dict[int, float], wins: Dict[int, int], out_path: str) -> None:
    rows = []
    for p, r in ratings.items():
        rows.append({
            'player': p,
            'bt_rating': r,
            'elo': convert_to_elo(r),
            'wins': wins.get(p, 0),
        })
    rows.sort(key=lambda x: x['bt_rating'], reverse=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['player','bt_rating','elo','wins'])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Dirty Graph BT-gradient rating calculation for chess-like games.")
    ap.add_argument('-f', '--file', type=str, required=True,
                    help='CSV with columns: game_no, player_first, player_second, result (w/l for player_first)')
    ap.add_argument('-l', '--learning-rate', type=float, default=0.2,
                    help='Learning rate η for rating updates (default: 0.2)')
    ap.add_argument('-o', '--output-csv', type=str, default='dirty_graph_ratings.csv',
                    help='Output CSV for ratings (default: dirty_graph_ratings.csv)')
    ap.add_argument('--phase2-iters', type=int, default=5, help='Max iterations for Phase 2 cleaning (default: 5)')
    ap.add_argument('--verbose-every', type=int, default=0,
                    help='Print progress every N games (0=off)')
    args = ap.parse_args()

    games = load_games_from_csv(args.file)
    phase1_r, final_r, wins = train_with_dirty_edges(
        games,
        learning_rate=args.learning_rate,
        phase2_max_iters=args.phase2_iters,
        verbose_every=args.verbose_every
    )
    save_ratings_to_csv(final_r, wins, args.output_csv)

    print("\nPhase 1 Ratings:")
    print_ratings(phase1_r, wins)
    print("\nFinal Ratings:")
    print_ratings(final_r, wins)


if __name__ == '__main__':
    main()
