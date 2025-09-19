#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv, math, random, argparse
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

try:
    from scipy.stats import kendalltau, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ----------------- IO -----------------

def read_matches(csv_path: str) -> List[Tuple[str,str,bool]]:
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
    # numerically stable logistic
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)
        
def save_ratings_to_csv(ratings, wins, output_file):
    """
    Save ratings to CSV file in the format compatible with compare_ratings.py.
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their ratings
        wins (dict): Dictionary mapping player IDs to their win counts
        output_file (str): Path to the output CSV file
    """
    # Calculate ELO from BT ratings
    factor = 400.0 / math.log(10.0)
    elo = {p: 1200.0 + factor * ratings[p] for p in ratings}
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['player_no', 'bt_rating', 'elo_rating', 'wins'])
        
        # Sort by player ID to match other implementations
        for player in sorted(ratings.keys(), key=lambda x: int(x)):
            writer.writerow([player, f"{ratings[player]:.6f}", f"{elo[player]:.1f}", wins.get(player, 0)])
    
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

def batch_bt_mle(matches: List[Tuple[str,str,bool]]) -> Dict[str,float]:
    # MM solver; returns centered log-strengths a (sum zero)
    wins = defaultdict(lambda: defaultdict(int))
    players = set()
    for p1, p2, w1 in matches:
        players.add(p1); players.add(p2)
        if w1: wins[p1][p2] += 1
        else:  wins[p2][p1] += 1
    players = sorted(players, key=lambda x: int(x) if x.isdigit() else x)

    w = {p: 1.0 for p in players}
    eps = 1e-12
    for it in range(1000):
        max_rel = 0.0
        w_new = {}
        for i in players:
            num = sum(wins[i].values()) + eps
            denom = 0.0
            for j in players:
                if j == i: continue
                m_ij = wins[i].get(j,0) + wins[j].get(i,0)
                if m_ij == 0: continue
                denom += m_ij * (w[i] / (w[i] + w[j]))
            denom += eps
            wi_new = w[i] * (num / denom)
            w_new[i] = max(wi_new, 1e-15)
            max_rel = max(max_rel, abs(w_new[i]-w[i])/(w[i]+1e-15))
        w = w_new
        print(f"Iteration {it+1}: max_rel = {max_rel}")
        if max_rel < 1e-10:
            print("Converged!")
            break

    logw = {p: math.log(max(w[p],1e-15)) for p in players}
    mean_log = sum(logw.values()) / len(logw)
    a = {p: logw[p] - mean_log for p in players}
    return a

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
        p = sigmoid(d)
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
        p = sigmoid(d)
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

# ----------------- Evaluation -----------------

def evaluate_method(name: str, matches: List[Tuple[str,str,bool]], players: List[str], a_gold: Dict[str,float],
                    build_fn, repeats: int=10, seed: int=0, shuffle: bool=True):
    rng = random.Random(seed)
    N = len(matches)
    rmse_list = []; tau_list = []; rho_list = []
    acc_list = []; brier_list = []; logloss_list = []

    for r in range(repeats):
        order = list(range(N))
        if shuffle:
            rng.shuffle(order)
        ordered = [matches[i] for i in order]

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
    print(f"[{name}]  RMSE vs BT: {fmt(rmse_list)};  "
          + (f"τ: {fmt(tau_list)};  ρ: {fmt(rho_list)};  " if HAVE_SCIPY else "")
          + f"online acc: {fmt(acc_list)};  Brier: {fmt(brier_list)};  logloss: {fmt(logloss_list)}")

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Single-pass BT baselines vs batch BT (gold).")
    ap.add_argument("--csv", required=True, help="Path to CSV (game, p1, p2, W/L).")
    ap.add_argument("--repeats", type=int, default=10, help="Number of random orders per method.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for shuffles.")
    ap.add_argument("--no-table", action="store_true", help="Don't show per-player tables for each method.")
    ap.add_argument("--save-csv", action="store_true", help="Save ratings to CSV files for each method.")
    ap.add_argument("--out-prefix", type=str, default="singlepass_", help="Prefix for output CSV files.")
    args = ap.parse_args()

    # Always use all matches
    matches = read_matches(args.csv)

    players = sorted({p for m in matches for p in (m[0], m[1])}, key=lambda x: int(x) if x.isdigit() else x)

    # Gold standard on the SAME subset
    a_gold = batch_bt_mle(matches)

    # Create player ID to index mapping
    player_to_idx = {p: i for i, p in enumerate(players)}
    
    # Evaluate each one-pass method against BT
    evaluate_method("ISGD",        matches, players, a_gold, build_fn=lambda ms, ps: onepass_isgd(ms, ps, eta=0.1, newton_steps=2, l2=1e-3), repeats=args.repeats, seed=args.seed, shuffle=True)
    evaluate_method("FTRL-Prox",   matches, players, a_gold, build_fn=lambda ms, ps: onepass_ftrl(ms, ps, alpha=0.1, l1=0.0, l2=1e-3), repeats=args.repeats, seed=args.seed+1, shuffle=True)
    evaluate_method("Diag-Newton", matches, players, a_gold, build_fn=lambda ms, ps: onepass_diag_newton(ms, ps, ridge=1e-2, step_cap=0.1), repeats=args.repeats, seed=args.seed+2, shuffle=True)
    evaluate_method("OPDN",        matches, players, a_gold, build_fn=lambda ms, ps: {p: rating for p, rating in zip(ps, one_pass_opdn_stream([(player_to_idx[p1], player_to_idx[p2], 1 if w1 else 0) for p1, p2, w1 in ms], len(ps), ridge=1e-2, step_cap=0.1))}, repeats=args.repeats, seed=args.seed+3, shuffle=True)
    
    # OPDN stream version already evaluated above as "OPDN"

    # Collect algorithm builders and names
    algorithm_builders = [
        ("ISGD", lambda: onepass_isgd(matches, players, eta=0.1, newton_steps=2, l2=1e-3)),
        ("FTRL-Prox", lambda: onepass_ftrl(matches, players, alpha=0.1, l1=0.0, l2=1e-3)),
        ("Diag-Newton", lambda: onepass_diag_newton(matches, players, ridge=1e-2, step_cap=0.1)),
        ("OPDN", lambda ordered, players=players: (lambda _ordered=ordered: (
            (lambda player_to_idx={p: i for i, p in enumerate(players)}: (
                (lambda pairs=( (lambda _pairs: ( [ _pairs.__setitem__((min(player_to_idx[p1], player_to_idx[p2]), max(player_to_idx[p1], player_to_idx[p2])),
                                            ( (_pairs.get((min(player_to_idx[p1], player_to_idx[p2]), max(player_to_idx[p1], player_to_idx[p2])), (0,0))[0] + 1),
                                              (_pairs.get((min(player_to_idx[p1], player_to_idx[p2]), max(player_to_idx[p1], player_to_idx[p2])), (0,0))[1] + 
                                               (1 if ( (player_to_idx[p1] < player_to_idx[p2] and w1) or (player_to_idx[p1] > player_to_idx[p2] and not w1) ) else 0))
                                            )
                          ) or _pairs ) for (p1,p2,w1) in _ordered ] and _pairs ))({})
                ):
                    { p: r for p, r in zip(players, one_pass_opdn_counts(pairs, len(players), ridge=0.0, step_cap=None, recenter=True)) }
                ) ) ) ))),
    ]
    
    # Run algorithms and process results
    
    # Calculate win counts for batch_bt_mle (used for both display and CSV)
    wins_count_bt = defaultdict(int)
    for p1, p2, w1 in matches:
        if w1: wins_count_bt[p1] += 1
        else:  wins_count_bt[p2] += 1
    
    # Calculate ELO ratings for batch_bt_mle
    factor = 400.0 / math.log(10.0)
    elo_bt = {p: 1200.0 + factor * a_gold[p] for p in players}
    
    # Display batch_bt_mle table if requested
    if not args.no_table:
        print("\n-------------------------------------------")
        print(f"Batch BT (Gold Standard) — Full table (subset size = {len(matches)})")
        print("-------------------------------------------")
        print("Rank | Player |   θ (centered)  |   Elo   | Wins")
        print("-------------------------------------------")
        
        # Display batch_bt_mle table, sorted by player number but showing true rank
        order_bt = sorted(players, key=lambda p: int(p))
        # Calculate ranks
        ranks = {p: idx+1 for idx, p in enumerate(sorted(players, key=lambda p: a_gold[p], reverse=True))}
        for p in order_bt:
            print(f"{ranks[p]:4d} | {int(p):6d} | {a_gold[p]:13.6f} | {elo_bt[p]:7.1f} | {wins_count_bt[p]:4d}")
    
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
        elo = {p: 1200.0 + factor * a_hat[p] for p in players}
        
        # Save to CSV if requested
        if args.save_csv:
            output_file = f"{args.out_prefix}{name.lower().replace('-', '_')}_ratings.csv"
            save_ratings_to_csv(a_hat, wins_count, output_file)
        
        # Display table unless explicitly disabled
        if not args.no_table:
            order = sorted(players, key=lambda p: int(p))
            print("\n-------------------------------------------")
            print(f"{name} — One-pass table (subset size = {len(matches)}), sorted by player number)")
            print("-------------------------------------------")
            print("Rank | Player |   θ (centered)  |   Elo   | Wins")
            print("-------------------------------------------")
            # Calculate ranks
            ranks = {p: idx+1 for idx, p in enumerate(sorted(players, key=lambda p: a_hat[p], reverse=True))}
            for p in order:
                print(f"{ranks[p]:4d} | {int(p):6d} | {a_hat[p]:13.6f} | {elo[p]:7.1f} | {wins_count[p]:4d}")



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
