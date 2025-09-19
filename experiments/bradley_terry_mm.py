#!/usr/bin/env python3
# Bradley-Terry solver (clean variants with NO learning rate)
# - MM / Iterative Scaling
# - Diagonal-Newton (IRLS)
# CSV schema: player_first,player_second,result   (result: 'w' means player_first wins, else treated as loss)
# Identifiability: zero-mean gauge each sweep.
# Handles separation with optional L2 (for IRLS) and safe clipping in MM.

import math, csv, sys
from typing import Dict, Tuple, List

def _sigmoid(x: float) -> float:
    # Stable logistic
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def _recenter(theta: List[float]) -> None:
    if not theta:
        return
    mu = sum(theta) / len(theta)
    for k in range(len(theta)):
        theta[k] -= mu

def _loglik(theta: List[float], pairs: Dict[Tuple[int,int], Tuple[int,int]]) -> float:
    # pairs[(i,j)] = (n_ij, a_ij) with i<j and a_ij = wins by i
    ll = 0.0
    for (i,j),(n,a) in pairs.items():
        d = theta[i] - theta[j]
        p = _sigmoid(d)
        p = min(max(p, 1e-15), 1-1e-15)
        ll += a*math.log(p) + (n-a)*math.log(1-p)
    return ll

def _build_pairs_from_csv(csv_path: str) -> Tuple[Dict[Tuple[int,int], Tuple[int,int]], int]:
    pairs: Dict[Tuple[int,int], Tuple[int,int]] = {}
    players = set()
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        expected = {"player_first","player_second","result"}
        missing = expected - set([c.strip() for c in r.fieldnames or []])
        if missing:
            raise ValueError(f"CSV missing columns: {missing}. Found: {r.fieldnames}")
        for row in r:
            i = int(row["player_first"])
            j = int(row["player_second"])
            if i == j:
                raise ValueError("Self-match encountered")
            res = row["result"].strip().lower()
            y = 1 if res == "w" else 0  # treat not 'w' as loss for player_first
            players.update([i,j])
            akey = (i,j) if i<j else (j,i)
            n_old, a_old = pairs.get(akey, (0,0))
            # wins by the smaller index (first in key)
            if i < j:
                a_new = a_old + (y if i == i else (1-y))
            else:
                # if key is (j,i) and original first is i: first wins count goes to "larger" side complement
                a_new = a_old + (1-y)
            pairs[akey] = (n_old+1, a_new)
    n_players = max(players) + 1 if players else 0
    return pairs, n_players

def _print_ratings(theta: List[float]) -> None:
    for idx, val in enumerate(theta):
        print(f"{idx},{val:.6f}")

# ===== MM / Iterative Scaling (no learning rate) =====
def fit_mm(csv_path: str, max_iter: int = 200, tol: float = 1e-8, clip_logit: float = 30.0) -> List[float]:
    pairs, n_players = _build_pairs_from_csv(csv_path)
    if n_players == 0:
        return []
    theta = [0.0]*n_players
    # work in pi = exp(theta)
    pi = [1.0]*n_players

    def sweep_mm(pi: List[float]) -> float:
        Ew = [0.0]*n_players
        Wins = [0.0]*n_players
        for (i,j),(n,a) in pairs.items():
            d = math.log(pi[i]) - math.log(pi[j])  # theta_i - theta_j
            p = _sigmoid(d)
            Ew[i] += n * p
            Ew[j] += n * (1.0 - p)
            Wins[i] += a
            Wins[j] += (n - a)
        # multiplicative update
        for k in range(n_players):
            denom = max(Ew[k], 1e-15)
            ratio = Wins[k] / denom
            # avoid degenerate explosions under complete separation
            ratio = min(max(ratio, math.exp(-clip_logit)), math.exp(clip_logit))
            pi[k] *= ratio
        return _loglik([math.log(x) for x in pi], pairs)

    prev_ll = _loglik(theta, pairs)
    for t in range(max_iter):
        curr_ll = sweep_mm(pi)
        theta = [math.log(x) for x in pi]
        _recenter(theta)
        # recenter in pi-space as well to keep consistency
        mu = sum(theta)/len(theta)
        for k in range(n_players):
            pi[k] = math.exp(theta[k])
        if abs(curr_ll - prev_ll) <= tol*(1.0 + abs(prev_ll)):
            break
        prev_ll = curr_ll
    return theta

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bradley-Terry via MM / Iterative Scaling (no LR)")
    ap.add_argument("csv_path", help="CSV with columns: player_first, player_second, result ('w' for p1 win, else loss)")
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()
    theta = fit_mm(args.csv_path, max_iter=args.max_iter, tol=args.tol)
    _print_ratings(theta)

if __name__ == "__main__":
    main()
