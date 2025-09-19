#!/usr/bin/env python3
"""
Dirty Graph Ratings (BT-gradient, dirty-edge, counts-weighted)

What changed (minimal, surgical):
- Edge.contribution1/2 now cache EXPECTED COUNTS (E1, E2), not gradient pieces.
- Node gains two scalars: actual (wins) and expected (sum of cached expected counts).
- When a dirty edge is touched, we "remove old / add new" expected counts for BOTH endpoints,
  then take a per-node gradient step: rating += eta * (actual - expected).
- Scheduling (dirty-for-one-end), two-phase flow, normalization, CSV I/O remain the same.

This stays fully incremental (O(m) per dirty neighborhood) and matches BT's counts-weighted gradient.
"""

import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, Tuple, List


class Edge:
    """An edge represents games between two players (player1 < player2)."""
    def __init__(self, player1: int, player2: int):
        self.player1 = min(player1, player2)  # canonical ordering
        self.player2 = max(player1, player2)

        # Observed counts
        self.wins1 = 0  # wins for player1 over player2
        self.wins2 = 0  # wins for player2 over player1

        # Cached EXPECTED COUNTS (not ratios, not gradients)
        # Invariant: exp1 + exp2 == total_games (up to FP error)
        self.contribution1 = 0.0  # expected wins credited to player1
        self.contribution2 = 0.0  # expected wins credited to player2

        # Per-endpoint dirty flags (scheduling only)
        self.dirty_for1 = True
        self.dirty_for2 = True

    def __repr__(self):
        n = self.total_games()
        return (f"Edge({self.player1}-{self.player2}, W1:{self.wins1}, W2:{self.wins2}, "
                f"E1:{self.contribution1:.3f}, E2:{self.contribution2:.3f}, "
                f"D1:{self.dirty_for1}, D2:{self.dirty_for2}, N:{n})")

    def add_game_result(self, winner: int, loser: int):
        """Add a game result and mark dirty for both endpoints."""
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


class Node:
    """A node represents a player with rating and incident edges."""
    def __init__(self, player_id: int):
        self.id = player_id
        self.rating = 0.0                         # BT log-rating
        self.edges: Dict[int, Edge] = {}          # opponent_id -> Edge

        # NEW: totals used in BT gradient (counts, not ratios)
        self.actual = 0                           # integer: total wins for this player
        self.expected = 0.0                       # float: sum of expected wins from incident edges

    def __repr__(self):
        return f"Node({self.id}, r:{self.rating:.4f}, A:{self.actual}, E:{self.expected:.3f}, deg:{len(self.edges)})"

    def add_edge(self, edge: Edge, opponent_id: int):
        self.edges[opponent_id] = edge

    def get_edge(self, opponent_id: int):
        return self.edges.get(opponent_id, None)


# -------------------- IO helpers --------------------

def load_games_from_csv(file_path: str) -> List[dict]:
    """
    Load game data from a CSV with columns:
    - game_no (int), player_first (int), player_second (int), result ('w' if player_first won, 'l' otherwise)
    """
    games = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            games.append({
                'game_no': int(row['game_no']),
                'player_first': int(row['player_first']),
                'player_second': int(row['player_second']),
                'result': row['result'].strip().lower(),  # 'w' or 'l'
            })
    return games


def get_player_list(games: List[dict]) -> List[int]:
    players = set()
    for g in games:
        players.add(g['player_first'])
        players.add(g['player_second'])
    return sorted(players)


def initialize_graph(players: List[int]) -> Dict[int, Node]:
    return {p: Node(p) for p in players}


# -------------------- BT math helpers --------------------

def compute_expected_score(r1: float, r2: float) -> float:
    """σ(r1 - r2) with simple guard for stability."""
    x = max(min(r1 - r2, 30.0), -30.0)  # clamp to avoid overflow
    return 1.0 / (1.0 + math.exp(-x))


def compute_edge_contributions(edge: Edge, node1: Node, node2: Node) -> Tuple[float, float]:
    """
    Return EXPECTED COUNTS (E1, E2) for this edge under CURRENT ratings.
    E1 = n * σ(θ1 − θ2), E2 = n − E1. If no games: (0, 0).
    node1 corresponds to edge.player1, node2 to edge.player2.
    """
    n = edge.total_games()
    if n == 0:
        return 0.0, 0.0
    p1 = compute_expected_score(node1.rating, node2.rating)
    E1 = n * p1
    E2 = n - E1
    return E1, E2


def refresh_edge_expected(edge: Edge, node1: Node, node2: Node) -> None:
    """
    Maintain the invariant that node.expected equals the sum of its edges' expected counts.
    Remove old E from both endpoints, recompute E with CURRENT ratings, add back.
    node1 must be the Node for edge.player1; node2 for edge.player2.
    """
    # Remove old expected from node totals
    node1.expected -= edge.contribution1
    node2.expected -= edge.contribution2

    # Recompute expected counts from current ratings
    E1, E2 = compute_edge_contributions(edge, node1, node2)

    # Store and add new
    edge.contribution1, edge.contribution2 = E1, E2
    node1.expected += E1
    node2.expected += E2


def normalize_ratings(nodes: Dict[int, Node]) -> None:
    """Mean-center ratings (identifiability; only differences matter)."""
    if not nodes:
        return
    mean_r = sum(n.rating for n in nodes.values()) / len(nodes)
    for n in nodes.values():
        n.rating -= mean_r


def convert_to_elo(bt_rating: float, anchor_bt: float = 0.0, anchor_elo: float = 1200.0) -> float:
    """ELO = anchor_elo + (400/ln 10) * (bt_rating - anchor_bt)."""
    return anchor_elo + (400.0 / math.log(10.0)) * (bt_rating - anchor_bt)


def print_ratings(ratings: Dict[int, float], wins: Dict[int, int]) -> None:
    elo = {p: convert_to_elo(r) for p, r in ratings.items()}
    sorted_players = sorted(ratings, key=lambda x: ratings[x], reverse=True)

    bt_vals = list(ratings.values())
    mean_bt = sum(bt_vals) / len(bt_vals) if bt_vals else 0.0
    median_bt = sorted(bt_vals)[len(bt_vals)//2] if bt_vals else 0.0
    min_bt = min(bt_vals) if bt_vals else 0.0
    max_bt = max(bt_vals) if bt_vals else 0.0
    var = sum((r - mean_bt)**2 for r in bt_vals) / len(bt_vals) if bt_vals else 0.0
    std_bt = math.sqrt(var)

    print("------------------------------------------------------------")
    print("Rank | Player | BT Rating    | ELO Rating | Wins ")
    print("------------------------------------------------------------")
    for rank, p in enumerate(sorted_players, 1):
        print(f"{rank:4d} | {p:6d} | {ratings[p]:11.6f} | {elo[p]:9.1f} | {wins.get(p,0):4d}")

    print("\nRating Statistics:")
    print(f"Mean Rating:   {mean_bt:.6f}")
    print(f"Median Rating: {median_bt:.6f}")
    print(f"Min Rating:    {min_bt:.6f}")
    print(f"Max Rating:    {max_bt:.6f}")
    print(f"Std Dev:       {std_bt:.6f}")


def save_ratings_to_csv(ratings: Dict[int, float], wins: Dict[int, int], file_path: str) -> None:
    with open(file_path, 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['player_no', 'bt_rating', 'elo_rating', 'wins'])
        for p, bt in sorted(ratings.items()):
            w.writerow([p, f"{bt:.6f}", f"{convert_to_elo(bt):.1f}", wins.get(p, 0)])


# -------------------- Core processing --------------------

def process_games(games: List[dict], learning_rate: float = 0.2,
                  phase2_max_iters: int = 20, verbose_every: int = 50) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, int]]:
    """
    Two-phase processing:
      Phase 1: stream games; for each game, repair dirty edges for the two players,
               refresh current edge, do gradient steps, then mark neighbors dirty.
      Phase 2: clean any leftover dirty edges for each player without further fan-out.
    Returns:
      (phase1_ratings, final_ratings, win_counts)
    """
    players = get_player_list(games)
    nodes = initialize_graph(players)
    edges: Dict[Tuple[int,int], Edge] = {}
    win_counts = {p: 0 for p in players}

    # -------- Phase 1 --------
    print("\nPhase 1: Processing individual games...")
    for idx, g in enumerate(games, 1):
        p1 = g['player_first']
        p2 = g['player_second']
        res = g['result']  # 'w' if player_first won, else 'l'

        # Create / get edge in canonical ordering
        key = (min(p1, p2), max(p1, p2))
        if key not in edges:
            e = Edge(p1, p2)
            edges[key] = e
            nodes[p1].add_edge(e, p2)
            nodes[p2].add_edge(e, p1)
        else:
            e = edges[key]

        # Winner/loser
        if res == 'w':
            winner, loser = p1, p2
        else:
            winner, loser = p2, p1

        # Update edge observed counts
        e.add_game_result(winner, loser)

        # Increment node-level "actual wins"
        nodes[winner].actual += 1
        win_counts[winner] += 1

        # --- Process all dirty edges for both endpoints (ONE HOP repair + gradient) ---
        for player in (p1, p2):
            node = nodes[player]
            # iterate over THIS player's incident edges
            for opp, edge in list(node.edges.items()):
                # check per-end dirty flag
                is_dirty_for_player = (edge.player1 == player and edge.dirty_for1) or \
                                      (edge.player2 == player and edge.dirty_for2)
                if not is_dirty_for_player:
                    continue

                # Refresh expected counts for BOTH endpoints (cheap; one sigmoid)
                n1 = nodes[edge.player1]
                n2 = nodes[edge.player2]
                refresh_edge_expected(edge, n1, n2)

                # Mark this side clean
                if edge.player1 == player:
                    edge.dirty_for1 = False
                else:
                    edge.dirty_for2 = False

                # Gradient step for THIS player only
                g_node = node.actual - node.expected
                node.rating += learning_rate * g_node

        # --- Now refresh the current game edge and update BOTH endpoints ---
        n1 = nodes[e.player1]
        n2 = nodes[e.player2]
        refresh_edge_expected(e, n1, n2)

        gw = nodes[winner].actual - nodes[winner].expected
        gl = nodes[loser].actual  - nodes[loser].expected
        nodes[winner].rating += learning_rate * gw
        nodes[loser].rating  += learning_rate * gl

        # Mark all OTHER edges of winner and loser dirty for their respective ends
        for player in (winner, loser):
            for opp, edge in nodes[player].edges.items():
                if (edge.player1 == winner and edge.player2 == loser) or (edge.player1 == loser and edge.player2 == winner):
                    continue  # skip the just-played edge
                if edge.player1 == player:
                    edge.dirty_for1 = True
                else:
                    edge.dirty_for2 = True

        if verbose_every and (idx % verbose_every == 0 or idx == len(games)):
            print(f"Processed {idx}/{len(games)} games")

    # Normalize & snapshot Phase 1 ratings
    normalize_ratings(nodes)
    phase1_ratings = {p: nodes[p].rating for p in players}
    print("\nPhase 1 Completed - Ratings after game processing:")
    print_ratings(phase1_ratings, win_counts)

    # -------- Phase 2: clean remaining dirty ends (no further fan-out) --------
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
    print(f"Starting Phase 2 with {len(players_with_dirty)} players having dirty edges")
    print(f"Total dirty ends: {sum(dirty_counts.values())}")

    iters = 0
    while players_with_dirty and iters < phase2_max_iters:
        iters += 1
        p = players_with_dirty.pop()
        node = nodes[p]
        cleaned = 0

        print(f"Phase 2 iteration {iters} - Player {p} with {dirty_counts[p]} dirty edges")

        for opp, edge in node.edges.items():
            is_dirty_for_p = (edge.player1 == p and edge.dirty_for1) or \
                             (edge.player2 == p and edge.dirty_for2)
            if not is_dirty_for_p:
                continue

            # Refresh expected counts for BOTH endpoints
            n1 = nodes[edge.player1]
            n2 = nodes[edge.player2]
            refresh_edge_expected(edge, n1, n2)

            # Mark this side clean
            if edge.player1 == p:
                edge.dirty_for1 = False
            else:
                edge.dirty_for2 = False
            dirty_counts[p] -= 1
            cleaned += 1

            # Gradient step for THIS player only
            g_node = node.actual - node.expected
            node.rating += learning_rate * g_node

        total_dirty = sum(dirty_counts.values())
        print(f"  Cleaned {cleaned} edges, remaining for {p}: {dirty_counts[p]}")
        print(f"  Total remaining dirty ends across all players: {total_dirty}")
        if dirty_counts[p] > 0:
            players_with_dirty.add(p)
        if total_dirty == 0:
            break

    print(f"Phase 2 completed after {iters} iterations")

    # Final normalize & return
    normalize_ratings(nodes)
    final_ratings = {p: nodes[p].rating for p in players}
    return phase1_ratings, final_ratings, win_counts


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Dirty Graph BT-gradient rating calculation for chess-like games.")
    ap.add_argument('-f', '--file', type=str, default='games.csv',
                    help='CSV file with columns: game_no, player_first, player_second, result (w/l)')
    ap.add_argument('-l', '--learning-rate', type=float, default=0.2,
                    help='Learning rate η for rating updates (default: 0.2)')
    ap.add_argument('-o', '--output-csv', type=str, default='dirty_graph_ratings.csv',
                    help='Output CSV for ratings (default: dirty_graph_ratings.csv)')
    ap.add_argument('--phase2-iters', type=int, default=20,
                    help='Max iterations of Phase 2 cleaning (default: 20)')
    ap.add_argument('--verbose-every', type=int, default=50,
                    help='Print progress every N games (default: 50)')

    args = ap.parse_args()

    print(f"Loading games from {args.file} ...")
    games = load_games_from_csv(args.file)
    print(f"Loaded {len(games)} games, {len(get_player_list(games))} unique players")

    phase1_r, final_r, wins = process_games(
        games,
        learning_rate=args.learning_rate,
        phase2_max_iters=args.phase2_iters,
        verbose_every=args.verbose_every
    )

    # Save final ratings
    save_ratings_to_csv(final_r, wins, args.output_csv)

    print("\nFinal Ratings:")
    print_ratings(final_r, wins)


if __name__ == '__main__':
    main()