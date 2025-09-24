#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, math, random, argparse
from collections import defaultdict
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

# Global compute counters for different algorithms
compute_counters = {
    'batch_bt': 0,
    'isgd': 0,
    'ftrl': 0,
    'diag_newton': 0,
    'opdn': 0,
    'dirtygraph': 0
}
import numpy as np

try:
    from scipy.stats import kendalltau, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Counter for Dirty Graph expected score calculations (legacy)
dg_expected_score_calls = 0

def reset_compute_counters():
    """Reset all compute counters to zero."""
    global compute_counters, dg_expected_score_calls
    for key in compute_counters:
        compute_counters[key] = 0
    dg_expected_score_calls = 0

def sigmoid_with_counter(x: float, algorithm: str) -> float:
    """Compute sigmoid with counter tracking for specified algorithm."""
    global compute_counters
    compute_counters[algorithm] += 1
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# ----------------- IO -----------------

def read_matches(filename: str) -> List[Tuple[str, str, bool]]:
    """Read CSV matches into (p1, p2, won_flag) format.""" 
    matches = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Auto-detect column names
            if 'player_first' in row:
                p1, p2 = row['player_first'], row['player_second']
            elif 'model_a' in row:
                p1, p2 = row['model_a'], row['model_b']
            else:
                raise ValueError(f"Unknown column format. Available columns: {list(row.keys())}")
            
            result = row['result'].strip().lower()
            won = (result == 'w')
            matches.append((p1, p2, won))
    return matches

def read_true_ratings(filename: str) -> Dict[str, float]:
    """Read true player ratings from CSV file."""
    ratings = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = str(row['player_id'])
            rating = float(row['rating'])
            ratings[player_id] = rating
    return ratings

def read_true_thetas(filename: str) -> Dict[str, float]:
    """Read true player Bradley-Terry thetas from CSV file."""
    thetas = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            player_id = str(row['player_id'])
            theta = float(row['theta'])
            thetas[player_id] = theta
    return thetas

def read_matches_old(csv_path: str) -> List[Tuple[str,str,bool]]:
    matches = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        first = next(rdr, None)
        def is_result(x): return isinstance(x, str) and x.strip().upper() in ('W','L')
        has_header = not is_result(first[3]) if (first and len(first)>=4) else True
        if not has_header and first:
            p1, p2, res = first[1].strip(), first[2].strip(), first[3].strip().upper()
            matches.append((p1, p2, res == 'W'))
        for row in rdr:
            if not row or len(row) < 4: continue
            p1, p2, res = row[1].strip(), row[2].strip(), row[3].strip().upper()
            if res not in ('W','L'): continue
            matches.append((p1, p2, res == 'W'))
    return matches

# ----------------- Utilities -----------------

def sigmoid(x: float) -> float:
    # numerically stable logistic (used by ISGD and FTRL)
    # Note: This will count for both ISGD and FTRL - we'll separate them by context in calls
    return sigmoid_with_counter(x, 'isgd')  # Default to ISGD for now

def sigmoid_ftrl(x: float) -> float:
    # sigmoid for FTRL algorithm
    return sigmoid_with_counter(x, 'ftrl')

def sigmoid_diag_newton(x: float) -> float:
    # sigmoid for diagonal newton algorithm
    return sigmoid_with_counter(x, 'diag_newton')

def print_compute_stats(algorithm_name: str):
    """Print compute statistics for the specified algorithm."""
    global compute_counters, dg_expected_score_calls
    
    algo_key = algorithm_name.lower().replace('-', '').replace(' ', '')
    if algo_key == 'dirtygraph':
        count = compute_counters['dirtygraph']
        print(f"[{algorithm_name}] Sigmoid computations: {count:,}")
    elif algo_key == 'opdn':
        count = compute_counters['opdn'] 
        print(f"[{algorithm_name}] Sigmoid computations: {count:,}")
    elif algo_key == 'isgd':
        count = compute_counters['isgd']
        print(f"[{algorithm_name}] Sigmoid computations: {count:,}")
    elif algo_key == 'ftrlprox':
        count = compute_counters['ftrl']
        print(f"[{algorithm_name}] Sigmoid computations: {count:,}")
    elif algo_key == 'diagnewton':
        count = compute_counters['diag_newton']
        print(f"[{algorithm_name}] Sigmoid computations: {count:,}")
    elif algo_key == 'batchbt' or algorithm_name == 'Batch BT':
        count = compute_counters['batch_bt']
        print(f"[{algorithm_name}] Sigmoid computations: {count:,}")
    else:
        print(f"[{algorithm_name}] Compute stats not tracked")
    
def get_total_compute_operations():
    """Get total sigmoid operations across all algorithms."""
    return sum(compute_counters.values())
        
def save_ratings_to_csv(ratings, wins, output_file):
    """
    Save ratings to CSV file in the format compatible with compare_ratings.py.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their ratings
        wins (dict): Dictionary mapping player IDs to their win counts
        output_file (str): Path to the output CSV file
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['player', 'rating', 'elo', 'wins'])
        for player in sorted(ratings.keys()):
            elo = 1400.0 + 400.0 * ratings[player] / math.log(10.0)
            writer.writerow([player, ratings[player], elo, wins[player]])
    
    print(f"Ratings saved to {output_file}")

def recenter(a: Dict[str,float]):
    mean_a = sum(a.values()) / len(a) if a else 0.0
    for k in a:
        a[k] -= mean_a

def loglike(a: Dict[str,float], matches: List[Tuple[str,str,bool]]) -> float:
    ll = 0.0
    for p1, p2, w1 in matches:
        d = a[p1] - a[p2]
        p = sigmoid(d)
        ll += math.log(p if w1 else (1.0 - p))
    return ll

def online_metrics(a: Dict[str,float], matches: List[Tuple[str,str,bool]], order: List[int]) -> Tuple[float,float,float]:
    # One streaming pass over matches[order], using current params (no extra updates here).
    # (This helper is unused in the main loop; we compute metrics inline to avoid extraneous passes.)
    n=0; acc=0; brier=0.0; logloss=0.0
    eps = 1e-15
    for idx in order:
        p1, p2, w1 = matches[idx]
        d = a[p1] - a[p2]
        p = sigmoid(d)
        y = 1.0 if w1 else 0.0
        acc += 1 if (p >= 0.5) == w1 else 0
        brier += (p - y)**2
        logloss += -( y*math.log(max(p,eps)) + (1-y)*math.log(max(1-p,eps)) )
        n += 1
    return (acc/n if n else 0.0, brier/n if n else 0.0, logloss/n if n else 0.0)

# ----------------- Batch BT (gold) -----------------

def batch_bt_mle(matches: List[Tuple[str,str,bool]], max_iters: int = 1000, threshold: float = 1e-10) -> Dict[str,float]:
    # MM solver; returns centered log-strengths a (sum zero)
    # Reset counter for batch BT
    compute_counters['batch_bt'] = 0
    
    wins = defaultdict(lambda: defaultdict(int))
    players = set()
    for p1, p2, w1 in matches:
        players.add(p1); players.add(p2)
        if w1: wins[p1][p2] += 1
        else:  wins[p2][p1] += 1
    players = sorted(players, key=lambda x: int(x) if x.isdigit() else x)

    w = {p: 1.0 for p in players}
    eps = 1e-12
    for it in range(max_iters):
        max_rel = 0.0
        w_new = {}
        for i in players:
            num = sum(wins[i].values()) + eps
            denom = 0.0
            for j in players:
                if j == i: continue
                m_ij = wins[i].get(j,0) + wins[j].get(i,0)
                if m_ij == 0: continue
                # Count this ratio computation as equivalent to sigmoid evaluation
                compute_counters['batch_bt'] += 1
                denom += m_ij * (w[i] / (w[i] + w[j]))
            denom += eps
            wi_new = w[i] * (num / denom)
            w_new[i] = max(wi_new, 1e-15)
            max_rel = max(max_rel, abs(w_new[i]-w[i])/(w[i]+1e-15))
        w = w_new
        if it % 10 == 0:
            print(f"Iteration {it}: max_rel = {max_rel}")
        if max_rel < threshold:
            print(f"Converged after {it} iterations!")
            break

    logw = {p: math.log(max(w[p],1e-15)) for p in players}
    mean_log = sum(logw.values()) / len(logw)
    a = {p: logw[p] - mean_log for p in players}
    return a

# ----------------- Dirty Graph data structures -----------------

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

# ----------------- Single-pass optimizers -----------------

def onepass_isgd(matches: List[Tuple[str,str,bool]], players: List[str], eta: float = 0.1, newton_steps: int = 2, l2: float = 1e-3) -> Dict[str,float]:
    """Implicit SGD for BT; one pass in given order; tiny L2; recenter at end."""
    a = {p: 0.0 for p in players}
    for p1, p2, w1 in matches:
        y = 1.0 if w1 else 0.0
        d = a[p1] - a[p2]

        # Solve Δ' = d + 2η( y - σ(Δ') ) via up to 'newton_steps' Newton iterations
        dp = d
        for _ in range(newton_steps):
            p = sigmoid(dp)
            g = dp - d - 2*eta*(y - p)         # f(dp) = dp - d - 2η(y - σ(dp)) = 0
            h = 1.0 - 2*eta*p*(1-p)            # f'(dp) = 1 - 2η σ'(dp)
            dp = dp - g / max(h, 1e-8)
        p = sigmoid(dp)
        grad = (y - p)

        a[p1] += eta * (grad - l2 * a[p1])
        a[p2] -= eta * (grad - l2 * a[p2])
    recenter(a)
    return a

def onepass_ftrl(matches: List[Tuple[str,str,bool]], players: List[str], alpha: float=0.1, l1: float=0.0, l2: float=1e-3) -> Dict[str,float]:
    """FTRL-Prox for BT/logistic with tiny L2. One pass. Recenter at end."""
    z = defaultdict(float)
    n = defaultdict(float)
    a = {p: 0.0 for p in players}

    def prox(z_i, n_i):
        # Closed form for logistic FTRL-Prox (no L1 by default)
        if l1 > 0 and abs(z_i) <= l1:
            return 0.0
        sign = -1.0 if z_i < 0.0 else 1.0
        return -(z_i - sign*l1) / ( (l2 + (math.sqrt(n_i)/alpha)) )

    for p1, p2, w1 in matches:
        y = 1.0 if w1 else 0.0
        d = a[p1] - a[p2]
        p = sigmoid_ftrl(d)
        # Gradients wrt a[p1], a[p2]
        g1 = -(y - p)
        g2 = +(y - p)
        # Update accumulators
        n1_old, n2_old = n[p1], n[p2]
        n[p1] += g1*g1
        n[p2] += g2*g2
        z[p1] += g1 - (math.sqrt(n[p1]) - math.sqrt(n1_old))/alpha * a[p1]
        z[p2] += g2 - (math.sqrt(n[p2]) - math.sqrt(n2_old))/alpha * a[p2]
        # Update weights via prox
        a[p1] = prox(z[p1], n[p1])
        a[p2] = prox(z[p2], n[p2])
    recenter(a)
    return a

def onepass_diag_newton(matches: List[Tuple[str,str,bool]], players: List[str],
                        ridge: float=1e-2, step_cap: float=0.1) -> Dict[str,float]:
    """Diagonal-Newton (online IRLS) with safeguards. One pass. Recenter at end."""
    a = {p: 0.0 for p in players}
    H = defaultdict(lambda: 1.0 + ridge)   # Start with solid curvature (1.0) + ridge

    for p1, p2, w1 in matches:
        y = 1.0 if w1 else 0.0
        d = a[p1] - a[p2]
        p = sigmoid_diag_newton(d)
        w = p*(1-p)                        # curvature
        # Per-parameter step with cap
        s1 = max(min((y - p) / H[p1], step_cap), -step_cap)
        s2 = max(min((y - p) / H[p2], step_cap), -step_cap)
        a[p1] += s1
        a[p2] -= s2
        H[p1] += w + ridge
        H[p2] += w + ridge
    recenter(a)
    return a

def compute_dg_expected_score(r1: float, r2: float) -> float:
    """Compute sigmoid(r1 - r2) with clamping for stability."""
    global dg_expected_score_calls
    dg_expected_score_calls += 1
    x = max(min(r1 - r2, 100.0), -100.0)  # Clamping to ±100
    return sigmoid_with_counter(x, 'dirtygraph')

def refresh_edge_expected(edge: Edge, node1: Node, node2: Node) -> None:
    """Update expected scores for an edge between two nodes."""
    # Remove old expected scores
    node1.expected -= edge.contribution1
    node2.expected -= edge.contribution2
    
    # Compute new expected scores
    n = edge.total_games()
    if n == 0:
        edge.contribution1, edge.contribution2 = 0.0, 0.0
        return
        
    p1 = compute_dg_expected_score(node1.rating, node2.rating)
    E1 = n * p1
    E2 = n - E1
    
    # Update contributions
    edge.contribution1, edge.contribution2 = E1, E2
    node1.expected += E1
    node2.expected += E2

def process_dirty_edges_for_player(nodes, player_idx, learning_rate=0.01, edge_cap=None):
    """Process dirty edges for a player. Return True if any were dirty.
    
    Args:
        nodes: Dictionary of player nodes
        player_idx: Index of player to process
        learning_rate: Learning rate for rating updates
        edge_cap: Maximum number of dirty edges to process (None = no limit)
    """
    node = nodes[player_idx]
    any_dirty = False
    edges_processed = 0
    
    # Process dirty edges (with optional cap)
    for opp_idx, edge in node.edges.items():
        # Check edge cap
        if edge_cap is not None and edges_processed >= edge_cap:
            break
            
        opp_node = nodes[opp_idx]
        
        # Check if edge is dirty for this player
        if ((edge.player1 == player_idx and edge.dirty_for1) or 
            (edge.player2 == player_idx and edge.dirty_for2)):
            any_dirty = True
            edges_processed += 1
            
            # Update expected scores
            if edge.player1 == player_idx:
                refresh_edge_expected(edge, node, opp_node)
                edge.dirty_for1 = False  # Mark as clean for this player
            else:
                refresh_edge_expected(edge, opp_node, node)
                edge.dirty_for2 = False  # Mark as clean for this player
    
    """
    # Update player rating if any edges were dirty
    if any_dirty:
        # ONLY accumulate gradient from edges refreshed in THIS call
        processed = []  # list of (A_node, E_node) for edges we just cleaned

        for opp_idx, edge in node.edges.items():
            # is this edge dirty for THIS player?
            is_dirty_for_player = (
                (edge.player1 == player_idx and edge.dirty_for1) or
                (edge.player2 == player_idx and edge.dirty_for2)
            )
            if not is_dirty_for_player:
                continue
            
            # cap-K guard if you use one
            if edge_cap is not None and len(processed) >= edge_cap:
                break

            # remove old expected contribution for THIS node
            if edge.player1 == player_idx:
                node.expected -= edge.contribution1
            else:
                node.expected -= edge.contribution2

            # recompute expected on this edge
            opp_node = nodes[opp_idx]
            n = edge.total_games()
            if n > 0:
                p_node = compute_dg_expected_score(node.rating, opp_node.rating)
                E_node = n * p_node
            else:
                E_node = 0.0

            # install new expected + clear dirtiness for THIS side
            if edge.player1 == player_idx:
                edge.contribution1 = E_node
                edge.dirty_for1 = False
                A_node = edge.wins1
            else:
                edge.contribution2 = E_node
                edge.dirty_for2 = False
                A_node = edge.wins2

            node.expected += E_node
            processed.append((A_node, E_node))

        if processed:
            local_grad = sum(A - E for (A, E) in processed)
            node.rating += learning_rate * local_grad

        # mark OUTBOUND edges dirty for neighbors (since this node changed)
        for opp_idx, edge in node.edges.items():
            if edge.player1 == player_idx:
                edge.dirty_for2 = True
            else:
                edge.dirty_for1 = True




    """
    # Update player rating if any edges were dirty
    if any_dirty:
        # Simple gradient approach (original):
        gradient = node.actual - node.expected
        node.rating += learning_rate * gradient
        """
        # FIX: accumulate gradient only from the edges we refreshed now
        local_grad = 0.0
        for opp_idx, edge in node.edges.items():
            # only count edges we marked clean for *this* player in this call
            if edge.player1 == player_idx and not edge.dirty_for1:
                A_node = edge.wins1
                E_node = edge.contribution1
                local_grad += (A_node - E_node)
            elif edge.player2 == player_idx and not edge.dirty_for2:
                A_node = edge.wins2
                E_node = edge.contribution2
                local_grad += (A_node - E_node)

        node.rating += learning_rate * local_grad
        """
        # Mark all edges to neighbors as dirty (for the neighbor's side)
        for opp_idx, edge in node.edges.items():
            if edge.player1 == player_idx:
                edge.dirty_for2 = True  # Mark as dirty for opponent
            else:
                edge.dirty_for1 = True  # Mark as dirty for opponent
    
    return any_dirty

def onepass_dirty_graph(matches: List[Tuple[str,str,bool]], players: List[str], 
                        learning_rate: float = 0.01, phase2_iters: int = 5, edge_cap: int = None) -> Dict[str,float]:
    """Dirty Graph algorithm for Bradley-Terry, adapted for single-pass framework."""
    # Reset counter
    global dg_expected_score_calls
    dg_expected_score_calls = 0
    
    # Create player index mapping
    player_to_idx = {p: i for i, p in enumerate(players)}
    idx_to_player = {i: p for i, p in enumerate(players)}
    
    # Initialize graph
    nodes = {i: Node(i) for i in range(len(players))}
    
    # Phase 1: Stream games
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
        process_dirty_edges_for_player(nodes, winner, learning_rate, edge_cap)
        process_dirty_edges_for_player(nodes, loser, learning_rate, edge_cap)
    
    # Phase 2: Clean remaining dirty edges
    for _ in range(phase2_iters):
        any_dirty = False
        for player_idx in range(len(players)):
            if process_dirty_edges_for_player(nodes, player_idx, learning_rate, edge_cap):
                any_dirty = True
        if not any_dirty:
            break
    
    # Extract final ratings and normalize
    ratings = {p: nodes[player_to_idx[p]].rating for p in players}
    mean_rating = sum(ratings.values()) / len(ratings)
    final_ratings = {p: r - mean_rating for p, r in ratings.items()}
    
    return final_ratings

# ----------------- Evaluation -----------------

def evaluate_method(name: str, matches: List[Tuple[str,str,bool]], players: List[str], a_gold: Dict[str,float], 
                   build_fn, repeats: int = 10, seed: int = 0, shuffle: bool = True):
    """Evaluate a rating method against the gold standard."""
    rng = random.Random(seed)
    N = len(matches)
    rmse_list = []; tau_list = []; rho_list = []
    acc_list = []; brier_list = []; logloss_list = []

    for r in range(repeats):
        order = list(range(N))
        if shuffle:
            rng.shuffle(order)
        ordered = [matches[i] for i in order]

        # Reset compute counters for this run
        reset_compute_counters()
        
        # Streaming online metrics during the single pass
        a_tmp = {p: 0.0 for p in players}  # ephemeral params for metrics
        # We'll compute online metrics INSIDE each builder as it updates; for simplicity,
        # we recompute metrics in a second pass using the final 'a' (gives upper bound).
        # To keep strictly one-pass, we instead compute metrics on-the-fly in each builder;
        # here we choose simplicity: evaluate with final 'a' on the same order.
        a_hat = build_fn(ordered, players)

        # RMSE vs gold
        v_gold = np.array([a_gold[p] for p in players])
        v_hat  = np.array([a_hat[p]  for p in players])
        rmse = float(np.sqrt(np.mean((v_hat - v_gold)**2)))
        rmse_list.append(rmse)

        # Rank agreements
        if HAVE_SCIPY:
            tau = float(kendalltau(v_gold, v_hat)[0])
            rho = float(spearmanr(v_gold, v_hat)[0])
            tau_list.append(tau); rho_list.append(rho)

        # Online predictive metrics on this order (using final a_hat as an approximation)
        eps = 1e-15
        n=0; acc=0; brier=0.0; logloss=0.0
        for (p1,p2,w1) in ordered:
            d = a_hat[p1] - a_hat[p2]
            p = sigmoid(d)
            y = 1.0 if w1 else 0.0
            acc += 1 if (p >= 0.5) == w1 else 0
            brier += (p - y)**2
            logloss += -( y*math.log(max(p,eps)) + (1-y)*math.log(max(1-p,eps)) )
            n += 1
        acc_list.append(acc/n)
        brier_list.append(brier/n)
        logloss_list.append(logloss/n)

    def fmt(xs): 
        if not xs: return "n/a"
        return f"{np.mean(xs):.6f} ± {np.std(xs):.6f}"
    print(f"[{name}]  RMSE vs Gold: {fmt(rmse_list)};  "
          + (f"τ: {fmt(tau_list)};  ρ: {fmt(rho_list)};  " if HAVE_SCIPY else "")
          + f"online acc: {fmt(acc_list)};  Brier: {fmt(brier_list)};  logloss: {fmt(logloss_list)}")
    
    # Display compute statistics
    print_compute_stats(name)

# ----------------- Main -----------------


def aggregate_pairs(ordered, players):
    """Build counts with canonical keys (i<j), a_ij = wins by i."""
    player_to_idx = {p: i for i, p in enumerate(players)}
    pairs = {}
    for p1, p2, w1 in ordered:
        i = player_to_idx[p1]; j = player_to_idx[p2]
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        n, a = pairs.get(key, (0, 0))
        win_is_min = (i < j and w1) or (i > j and not w1)
        pairs[key] = (n + 1, a + (1 if win_is_min else 0))
    return pairs, len(players)


def k_sweep_diag_newton_counts(pairs, n_players, sweeps=1, h_floor=1e-12):
    """Multi-sweep Jacobi-style diagonal-Newton over aggregated counts."""
    import math
    def _sig(x):
        return sigmoid_with_counter(x, 'opdn')

    theta = [0.0]*n_players
    for _ in range(sweeps):
        g = [0.0]*n_players
        h = [h_floor]*n_players
        for (i, j), (n, a) in pairs.items():
            d = theta[i] - theta[j]
            p = _sig(d)
            w = n * p * (1.0 - p)
            gi = a - n * p
            g[i] += gi; g[j] -= gi
            h[i] += w;  h[j] += w
        # unit Newton step, per-node (Jacobi)
        for k in range(n_players):
            theta[k] += g[k] / h[k]
        # recenter each sweep
        mean = sum(theta)/n_players
        for k in range(n_players):
            theta[k] -= mean
    return theta

def one_sweep_diag_newton_counts(pairs, n_players):
    """Single Jacobi-style diagonal-Newton sweep from theta=0 over aggregated counts."""
    return k_sweep_diag_newton_counts(pairs, n_players, sweeps=1)

def build_opdn(ordered, players, sweeps=1):
    """Counts-based diagonal-Newton (Jacobi) with 'sweeps' passes."""
    pairs, n_players = aggregate_pairs(ordered, players)
    theta = k_sweep_diag_newton_counts(pairs, n_players, sweeps=sweeps)
    return {p: r for p, r in zip(players, theta)}

def main():
    ap = argparse.ArgumentParser(description="Single-pass BT baselines vs batch BT (gold).")
    ap.add_argument("--csv", required=True, help="Path to CSV (game, p1, p2, W/L).")
    ap.add_argument("--repeats", type=int, default=10, help="Number of random orders per method.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for shuffles.")
    ap.add_argument("--no-table", action="store_true", help="Don't show per-player tables for each method.")
    ap.add_argument("--save-csv", action="store_true", help="Save ratings to CSV files for each method.")
    ap.add_argument("--out-prefix", type=str, default="singlepass_", help="Prefix for output CSV files.")
    ap.add_argument("--max-iterations", type=int, default=1000, help="Maximum iterations for batch BT MLE algorithm")
    ap.add_argument("--threshold", type=float, default=1e-10, help="Convergence threshold for batch BT MLE algorithm")
    ap.add_argument("--edge-cap", type=int, default=10, help="Edge cap for DirtyGraph algorithm (default: 10)")
    ap.add_argument("--opdn-passes", type=int, default=1, help="Number of passes through the data for OPDN algorithm (default: 1)")
    ap.add_argument("--true-ratings", type=str, default=None, help="Path to CSV file with true player ratings for comparison (default: None, uses Batch BT)")
    ap.add_argument("--elo-comparison", action="store_true", help="Compare algorithms directly in ELO space instead of Bradley-Terry space")
    ap.add_argument("--bt-comparison", action="store_true", help="Compare algorithms directly in Bradley-Terry theta space")
    args = ap.parse_args()

    # Always use all matches
    matches = read_matches(args.csv)
    
    # Get player list
    players = sorted({p for m in matches for p in (m[0], m[1])}, key=lambda x: int(x) if x.isdigit() else x)
    
    # Determine gold standard
    if args.true_ratings:
        # Check if we're dealing with BT thetas or ELO ratings
        with open(args.true_ratings, 'r') as f:
            header = f.readline().strip()
            is_theta_file = 'theta' in header.lower()
        
        if is_theta_file:
            # Use true Bradley-Terry thetas as gold standard
            true_thetas = read_true_thetas(args.true_ratings)
            a_gold = {p: true_thetas.get(p, 0.0) for p in players}
            # Center the thetas (sum to zero)
            mean_theta = sum(a_gold.values()) / len(a_gold)
            a_gold = {p: v - mean_theta for p, v in a_gold.items()}
            print(f"Using true Bradley-Terry thetas as gold standard (mean: {mean_theta:.3f})")
            gold_name = "True Thetas"
        else:
            # Use true ELO ratings as gold standard
            true_ratings = read_true_ratings(args.true_ratings)
            # Convert ELO ratings to Bradley-Terry log-strengths
            # ELO: P(i beats j) = 1/(1 + 10^((R_j - R_i)/400))
            # BT:  P(i beats j) = 1/(1 + exp(a_j - a_i))
            # So: a_i = (R_i / 400) * log(10) + constant
            ratings_values = [true_ratings.get(p, 1400) for p in players]  # Default 1400 if missing
            mean_rating = sum(ratings_values) / len(ratings_values)
            a_gold = {p: (true_ratings.get(p, 1400) / 400.0) * math.log(10) for p in players}
            # Center the ratings (sum to zero)
            mean_log = sum(a_gold.values()) / len(a_gold)
            a_gold = {p: v - mean_log for p, v in a_gold.items()}
            print(f"Using true ELO ratings as gold standard (mean: {mean_rating:.1f})")
            gold_name = "True Ratings"
    else:
        # Use Batch BT as gold standard
        a_gold = batch_bt_mle(matches, max_iters=args.max_iterations, threshold=args.threshold)
        print_compute_stats("Batch BT")
        gold_name = "Batch BT"

    # Create player ID to index mapping
    player_to_idx = {p: i for i, p in enumerate(players)}
    
    # Bradley-Terry comparison mode
    if args.bt_comparison and args.true_ratings:
        print("=== BRADLEY-TERRY Direct Comparison ===")
        true_thetas = read_true_thetas(args.true_ratings)
        
        # Test each algorithm and collect results
        algorithms = [
            ("Batch BT", lambda ms, ps: batch_bt_mle(ms, max_iters=args.max_iterations, threshold=args.threshold)),
            ("ISGD", lambda ms, ps: onepass_isgd(ms, ps, eta=0.1, newton_steps=2, l2=1e-3)),
            ("FTRL-Prox", lambda ms, ps: onepass_ftrl(ms, ps, alpha=0.1, l1=0.0, l2=1e-3)),
            ("Diag-Newton", lambda ms, ps: onepass_diag_newton(ms, ps, ridge=1e-2, step_cap=0.1)),
            ("OPDN", lambda ms, ps: build_opdn(ms, ps, sweeps=args.opdn_passes)),
            ("DirtyGraph", lambda ms, ps: onepass_dirty_graph(ms, ps, learning_rate=len(ps)/len(ms), phase2_iters=1, edge_cap=args.edge_cap))
        ]
        
        # Collect results from all algorithms
        results = {}
        
        for algo_name, build_fn in algorithms:
            # Reset counters
            reset_compute_counters()
            
            # Run algorithm
            a_pred = build_fn(matches, players)
            
            # Calculate RMSE in theta space (no conversion needed)
            rmse = 0.0
            count = 0
            for p in players:
                if p in true_thetas and p in a_pred:
                    rmse += (true_thetas[p] - a_pred[p]) ** 2
                    count += 1
            
            if count > 0:
                rmse = math.sqrt(rmse / count)
                print(f"[{algo_name}] Bradley-Terry Theta RMSE: {rmse:.6f}")
                print_compute_stats(algo_name)
            
            results[algo_name] = a_pred
        
        # Print comparison table
        print("\n" + "="*100)
        print("BRADLEY-TERRY THETA COMPARISON TABLE")
        print("="*100)
        print(f"{'Player':>6} | {'True θ':>8} | {'Batch BT':>8} | {'ISGD':>8} | {'FTRL':>8} | {'Diag-N':>8} | {'OPDN':>8} | {'DirtyG':>8}")
        print("-" * 100)
        
        for p in sorted(players, key=lambda x: int(x) if x.isdigit() else x):
            if p in true_thetas:
                row = f"{p:>6} | {true_thetas[p]:>8.3f}"
                for algo_name in ["Batch BT", "ISGD", "FTRL-Prox", "Diag-Newton", "OPDN", "DirtyGraph"]:
                    if algo_name in results and p in results[algo_name]:
                        row += f" | {results[algo_name][p]:>8.3f}"
                    else:
                        row += f" | {'N/A':>8}"
                print(row)
        
        print("="*100)
        
        return  # Exit early for BT comparison mode
    
    # ELO comparison mode
    if args.elo_comparison and args.true_ratings:
        print("=== ELO-to-ELO Direct Comparison ===")
        true_elo_ratings = read_true_ratings(args.true_ratings)
        
        # Test each algorithm and collect results
        algorithms = [
            ("Batch BT", lambda ms, ps: batch_bt_mle(ms, max_iters=args.max_iterations, threshold=args.threshold)),
            ("ISGD", lambda ms, ps: onepass_isgd(ms, ps, eta=0.1, newton_steps=2, l2=1e-3)),
            ("FTRL-Prox", lambda ms, ps: onepass_ftrl(ms, ps, alpha=0.1, l1=0.0, l2=1e-3)),
            ("Diag-Newton", lambda ms, ps: onepass_diag_newton(ms, ps, ridge=1e-2, step_cap=0.1)),
            ("OPDN", lambda ms, ps: build_opdn(ms, ps, sweeps=args.opdn_passes)),
            ("DirtyGraph", lambda ms, ps: onepass_dirty_graph(ms, ps, learning_rate=len(ps)/len(ms), phase2_iters=1, edge_cap=args.edge_cap))
        ]
        
        # Collect results from all algorithms
        results = {}
        
        for algo_name, build_fn in algorithms:
            # Reset counters
            reset_compute_counters()
            
            # Run algorithm
            a_pred = build_fn(matches, players)
            
            # Convert to ELO using existing conversion logic (a * 400/ln(10) + 1400)
            pred_elo = {p: a * 400.0 / math.log(10) + 1400.0 for p, a in a_pred.items()}
            
            # Calculate RMSE
            rmse = 0.0
            count = 0
            for p in players:
                if p in true_elo_ratings and p in pred_elo:
                    rmse += (true_elo_ratings[p] - pred_elo[p]) ** 2
                    count += 1
            
            if count > 0:
                rmse = math.sqrt(rmse / count)
                print(f"[{algo_name}] ELO RMSE: {rmse:.6f}")
                print_compute_stats(algo_name)
            
            results[algo_name] = pred_elo
        
        # Print comparison table
        print("\n" + "="*100)
        print("ELO COMPARISON TABLE")
        print("="*100)
        print(f"{'Player':>6} | {'True ELO':>8} | {'Batch BT':>8} | {'ISGD':>8} | {'FTRL':>8} | {'Diag-N':>8} | {'OPDN':>8} | {'DirtyG':>8}")
        print("-" * 100)
        
        for p in sorted(players, key=lambda x: int(x) if x.isdigit() else x):
            if p in true_elo_ratings:
                row = f"{p:>6} | {true_elo_ratings[p]:>8.1f}"
                for algo_name in ["Batch BT", "ISGD", "FTRL-Prox", "Diag-Newton", "OPDN", "DirtyGraph"]:
                    if algo_name in results and p in results[algo_name]:
                        row += f" | {results[algo_name][p]:>8.1f}"
                    else:
                        row += f" | {'N/A':>8}"
                print(row)
        
        print("="*100)
        
        return  # Exit early for ELO comparison mode
    
    # Regular evaluation against gold standard
    if args.true_ratings:
        # Include Batch BT in comparison when using true ratings
        evaluate_method("Batch BT", matches, players, a_gold, build_fn=lambda ms, ps: batch_bt_mle(ms, max_iters=args.max_iterations, threshold=args.threshold), repeats=args.repeats, seed=args.seed+5, shuffle=False)
    
    evaluate_method("ISGD",        matches, players, a_gold, build_fn=lambda ms, ps: onepass_isgd(ms, ps, eta=0.1, newton_steps=2, l2=1e-3), repeats=args.repeats, seed=args.seed, shuffle=True)
    evaluate_method("FTRL-Prox",   matches, players, a_gold, build_fn=lambda ms, ps: onepass_ftrl(ms, ps, alpha=0.1, l1=0.0, l2=1e-3), repeats=args.repeats, seed=args.seed+1, shuffle=True)
    evaluate_method("Diag-Newton", matches, players, a_gold, build_fn=lambda ms, ps: onepass_diag_newton(ms, ps, ridge=1e-2, step_cap=0.1), repeats=args.repeats, seed=args.seed+2, shuffle=True)
    evaluate_method("OPDN",        matches, players, a_gold, build_fn=lambda ms, ps: build_opdn(ms, ps, sweeps=args.opdn_passes), repeats=args.repeats, seed=args.seed+3, shuffle=True)
    evaluate_method("DirtyGraph", matches, players, a_gold, build_fn=lambda ms, ps: onepass_dirty_graph(ms, ps, learning_rate=len(ps)/len(ms), phase2_iters=1, edge_cap=args.edge_cap), repeats=args.repeats, seed=args.seed+4, shuffle=True)
    
    # OPDN stream version already evaluated above as "OPDN"

    # Collect algorithm builders and names
    algorithm_builders = [
        ("ISGD", lambda: onepass_isgd(matches, players, eta=0.1, newton_steps=2, l2=1e-3)),
        ("FTRL-Prox", lambda: onepass_ftrl(matches, players, alpha=0.1, l1=0.0, l2=1e-3)),
        ("Diag-Newton", lambda: onepass_diag_newton(matches, players, ridge=1e-2, step_cap=0.1)),
        ("OPDN", lambda: build_opdn(matches, players, sweeps=args.opdn_passes)),
        ("DirtyGraph", lambda: onepass_dirty_graph(matches, players, learning_rate=len(players)/len(matches), phase2_iters=1, edge_cap=args.edge_cap)),
    ]
    
    # Run algorithms and process results
    
    # Calculate win counts for batch_bt_mle (used for both display and CSV)
    wins_count_bt = defaultdict(int)
    for p1, p2, w1 in matches:
        if w1: wins_count_bt[p1] += 1
        else:  wins_count_bt[p2] += 1
    
    # Calculate ELO ratings for batch_bt_mle
    factor = 400.0 / math.log(10.0)
    elo_bt = {p: 1400.0 + factor * a_gold[p] for p in players}
    
    # Display batch_bt_mle table if requested
    if not args.no_table:
        print("\n-------------------------------------------")
        print(f"Batch BT (Gold Standard) — Full table (subset size = {len(matches)})")
        print("-------------------------------------------")
        print("Rank | Player |   θ (centered)  |   Elo   | Wins")
        print("-------------------------------------------")
        
        # Display batch_bt_mle table, sorted alphabetically but showing true rank
        order_bt = sorted(players)
        # Calculate ranks
        ranks = {p: idx+1 for idx, p in enumerate(sorted(players, key=lambda p: a_gold[p], reverse=True))}
        for p in order_bt:
            # Format player name to fit in column (truncate if too long)
            player_display = p[:15] if len(p) > 15 else p
            print(f"{ranks[p]:4d} | {player_display:15s} | {a_gold[p]:13.6f} | {elo_bt[p]:7.1f} | {wins_count_bt[p]:4d}")
    
    # Save batch_bt_mle to CSV if requested
    if args.save_csv:
        output_file = f"{args.out_prefix}batch_bt_ratings.csv"
        save_ratings_to_csv(a_gold, wins_count_bt, output_file)
    
    # Run each algorithm, generate results, and optionally show tables or save CSVs
    for name, builder in algorithm_builders:
        # This is where the actual algorithm runs - must run regardless of display options
        a_hat = builder()
        
        # Calculate win counts
        wins_count = defaultdict(int)
        for p1, p2, w1 in matches:
            if w1: wins_count[p1] += 1
            else:  wins_count[p2] += 1
        
        # Calculate ELO ratings
        factor = 400.0 / math.log(10.0)
        elo = {p: 1400.0 + factor * a_hat[p] for p in players}
        
        # Save to CSV if requested
        if args.save_csv:
            output_file = f"{args.out_prefix}{name.lower().replace('-', '_')}_ratings.csv"
            save_ratings_to_csv(a_hat, wins_count, output_file)
        
        # Display table unless explicitly disabled
        if not args.no_table:
            order = sorted(players)
            print("\n-------------------------------------------")
            print(f"{name} — One-pass table (subset size = {len(matches)}), sorted alphabetically by player ID)")
            print("-------------------------------------------")
            print("Rank | Player |   θ (centered)  |   Elo   | Wins")
            print("-------------------------------------------")
            # Calculate ranks
            ranks = {p: idx+1 for idx, p in enumerate(sorted(players, key=lambda p: a_hat[p], reverse=True))}
            for p in order:
                # Format player name to fit in column (truncate if too long)
                player_display = p[:15] if len(p) > 15 else p
                print(f"{ranks[p]:4d} | {player_display:15s} | {a_hat[p]:13.6f} | {elo[p]:7.1f} | {wins_count[p]:4d}")



# ===== OPDN: One-Pass Diagonal-Newton (symmetric step) =====
import math

def _opdn_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def _opdn_recenter(theta):
    mean = sum(theta) / len(theta) if theta else 0.0
    for k in range(len(theta)):
        theta[k] -= mean

def one_pass_opdn_counts(pairs, n_players, init=None, ridge: float = 0.0, step_cap: float | None = None, recenter: bool = True):
    """Single-sweep diagonal-Newton for aggregated counts.
    pairs: dict[(i,j)] -> (n_ij, a_ij) with i<j and a_ij = wins by i.
    Updates use equal-and-opposite steps s = (a_ij - n_ij * p) / (2*n_ij*p*(1-p) + 2*ridge).
    """
    eps = 1e-12
    theta = [0.0]*n_players if init is None else list(init)
    for (i, j), (n_ij, a_ij) in pairs.items():
        if i == j or n_ij <= 0:
            continue
        d = theta[i] - theta[j]
        p = _opdn_sigmoid(d)
        w = n_ij * p * (1.0 - p)
        g = (a_ij - n_ij * p)
        den = max(2.0*w + 2.0*ridge, eps)  # same denominator for both endpoints
        s = g / den
        if step_cap is not None:
            s = max(min(s, step_cap), -step_cap)
        theta[i] += s
        theta[j] -= s
    if recenter and n_players > 0:
        _opdn_recenter(theta)
    return theta

def one_pass_opdn_stream(matches, n_players, init=None, ridge: float = 0.0, step_cap: float | None = None, recenter: bool = True):
    """Single-sweep diagonal-Newton for an individual-duel stream.
    matches: iterable of (i, j, y) where y=1 if i wins, 0 if j wins.
    """
    eps = 1e-12
    theta = [0.0]*n_players if init is None else list(init)
    for (i, j, y) in matches:
        if i == j:
            continue
        d = theta[i] - theta[j]
        p = _opdn_sigmoid(d)
        w = p * (1.0 - p)
        g = (1.0 if y == 1 else 0.0) - p
        den = max(2.0*w + 2.0*ridge, eps)
        s = g / den
        if step_cap is not None:
            s = max(min(s, step_cap), -step_cap)
        theta[i] += s
        theta[j] -= s
    if recenter and n_players > 0:
        _opdn_recenter(theta)
    return theta



if __name__ == "__main__":
    main()
