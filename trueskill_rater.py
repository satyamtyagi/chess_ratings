#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrueSkill (win/loss only) CSV rater with approximate Elo conversion.

INPUT CSV FORMAT (no header expected by default):
    game_number, player1_id, player2_id, result
Where `result` is "W" if player1 won, "L" otherwise (i.e., player2 won).

WHY THE FORMULAS (plain-language recap):
- Each player's *skill* is modeled as Gaussian N(mu, sigma^2).
- A single game's *performance gap* is noisy; its variance is:
      c^2 = sigma_A^2 + sigma_B^2 + 2*beta^2
  (players' uncertainties + per-game performance noise for both sides).
- Standardize the skill gap: t = (mu_A - mu_B) / c.
  The pre-game probability A wins is Phi(t), where Phi is the standard normal CDF.
- After observing who won, we *condition on an inequality* ("A wins" means gap>0).
  The two scalars V(t) and W(t) from the truncated-normal algebra determine:
    • how much to move the means (via V)
    • how much to shrink the variances (via W)
- Updates (A wins case; B wins uses t -> -t and flips signs for the mean nudges):
    mu_A += (sigma_A^2 / c) * V(t)
    mu_B -= (sigma_B^2 / c) * V(t)
    sigma_A^2 *= (1 - (sigma_A^2 / c^2) * W(t))
    sigma_B^2 *= (1 - (sigma_B^2 / c^2) * W(t))

APPROXIMATE ELO CONVERSION (optional, for display only):
- Elo and TrueSkill use different link functions (logistic vs probit).
- For *small* rating gaps and mature players (similar uncertainty), probability in TS is
    P = Phi( (muA - muB) / (sqrt(2)*beta) ).
  Elo uses P = 1 / (1 + 10^(-ΔElo/400)).
- Matching the slopes near an even match yields an *approximate linear* mapping:
    Elo ≈ Elo0 + k * (mu - mu0), where k ≈ 196.1 / beta.
  With the common TS default beta = 25/6 ≈ 4.1667, k ≈ 47.1 Elo per μ.
  We'll expose mu0 and Elo0 so you can anchor 25 → 1500 by default.

USAGE:
    python trueskill_rater.py --csv matches.csv
    # with header row:
    python trueskill_rater.py --csv matches.csv --has-header
    # write results to a file:
    python trueskill_rater.py --csv matches.csv --out ratings.csv
    # show Elo using default anchor 25 -> 1500:
    python trueskill_rater.py --csv matches.csv --show-elo

Parameters (tune with flags):
- mu0=25, sigma0=25/3, beta=25/6, tau=0.0 (no drift). Change with flags if desired.
"""

import csv
import math
import argparse
from collections import defaultdict

# --------- Normal PDF/CDF utilities ---------

def phi(x: float) -> float:
    """Standard normal PDF: local 'height' of bell curve at x."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def Phi(x: float) -> float:
    """Standard normal CDF: probability a standard normal is <= x."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def invmills_ratio_for_win(t: float) -> float:
    """V(t) for 'A wins': V = phi(t) / (1 - Phi(t)).
    This is the inverse Mills ratio used as the step size for mean updates when A wins.
    """
    denom = 1.0 - Phi(t)
    if denom < 1e-12:  # numerical floor
        denom = 1e-12
    return phi(t) / denom

def W_from_V_t(V: float, t: float) -> float:
    """W(t) = V(t) * (V(t) + t). Controls how much we shrink variances (uncertainty)."""
    return V * (V + t)

# --------- TrueSkill (win/loss only) core ---------

class TrueSkillWL:
    def __init__(self, mu0=25.0, sigma0=25.0/3.0, beta=25.0/6.0, tau=0.0):
        # Model parameters:
        # beta: per-game performance noise (higher -> noisier games, smaller updates)
        # tau: skill drift per game (inflate variance before each match; 0 for stability)
        self.mu0 = float(mu0)
        self.s20 = float(sigma0)**2
        self.beta = float(beta)
        self.tau = float(tau)

        # Player state: maps of player_id -> (mu, var)
        self.mu = {}
        self.var = {}

    def ensure_player(self, pid):
        if pid not in self.mu:
            self.mu[pid] = self.mu0
            self.var[pid] = self.s20

    def update_match(self, p1, p2, p1_won: bool):
        """Update ratings for a 1v1 match.
        p1: player id for column 2 (Player1)
        p2: player id for column 3 (Player2)
        p1_won: True if column 4 is 'W', False if 'L'
        """
        self.ensure_player(p1); self.ensure_player(p2)

        # Inflate uncertainty by drift (optional; tau=0 leaves as-is)
        s2a = self.var[p1] + self.tau**2
        s2b = self.var[p2] + self.tau**2

        # Total uncertainty of the performance gap this game
        c2 = s2a + s2b + 2.0 * (self.beta**2)
        c = math.sqrt(c2)

        # Standardized skill gap at the decision boundary
        t = (self.mu[p1] - self.mu[p2]) / c

        # Compute V and W for the *observed* outcome
        if p1_won:
            V = invmills_ratio_for_win(t)
            W = W_from_V_t(V, t)
            sign = +1.0
        else:
            # If Player 2 won, reuse the "A wins" forms with t -> -t, then flip mean signs
            V = invmills_ratio_for_win(-t)
            W = W_from_V_t(V, -t)
            sign = -1.0

        # Mean updates: bigger V -> bigger move; scaled by each player's own uncertainty
        mu1 = self.mu[p1] + sign * (s2a / c) * V
        mu2 = self.mu[p2] - sign * (s2b / c) * V

        # Variance updates: informative games (bigger W) shrink uncertainty more
        s2a_new = s2a * (1.0 - (s2a / c2) * W)
        s2b_new = s2b * (1.0 - (s2b / c2) * W)

        # Commit updates with a tiny floor on variance for numerical safety
        self.mu[p1], self.mu[p2] = mu1, mu2
        self.var[p1] = max(s2a_new, 1e-9)
        self.var[p2] = max(s2b_new, 1e-9)

    def displayed(self, pid, k=3.0) -> float:
        """Conservative displayed rating: mu - k*sigma (k=3 is Xbox-style)."""
        self.ensure_player(pid)
        return self.mu[pid] - k * math.sqrt(self.var[pid])

# --------- Approximate Elo conversion ---------

def normalize_ratings(ratings, target_min=-0.25, target_max=0.31):
    """
    Normalize ratings to match Bradley-Terry range (approximately -0.25 to +0.31).
    
    Args:
        ratings (dict): Dictionary mapping player IDs to their ratings
        target_min (float): Target minimum value for normalized ratings (default: -0.3)
        target_max (float): Target maximum value for normalized ratings (default: 0.3)
    
    Returns:
        dict: Dictionary of normalized ratings scaled to match Bradley-Terry range
    """
    if not ratings:
        return {}
    
    # First center by subtracting the mean
    mean_rating = sum(ratings.values()) / len(ratings)
    centered = {player: rating - mean_rating for player, rating in ratings.items()}
    
    # Find min/max of centered ratings
    min_rating = min(centered.values())
    max_rating = max(centered.values())
    rating_range = max_rating - min_rating
    
    # Scale to target Bradley-Terry range
    target_range = target_max - target_min
    normalized = {}
    for player, rating in centered.items():
        if rating_range > 0:  # Avoid division by zero
            # Scale to [0,1] then to [target_min, target_max]
            normalized[player] = target_min + ((rating - min_rating) / rating_range) * target_range
        else:
            normalized[player] = 0.0
    
    return normalized

def convert_to_elo(bt_rating, elo_anchor=1200, bt_anchor=0):
    """
    Convert rating to ELO scale using Bradley-Terry formula.
    
    Args:
        bt_rating (float): Rating to convert (normalized TrueSkill μ)
        elo_anchor (int): ELO rating anchor point (default: 1200)
        bt_anchor (float): Rating anchor point (default: 0)
        
    Returns:
        float: Corresponding ELO rating
    """
    # Formula: ELO = elo_anchor + (400/ln(10)) * (BT_rating - bt_anchor)
    return elo_anchor + (400 / math.log(10)) * (bt_rating - bt_anchor)

# --------- CSV I/O and CLI ---------

def read_matches(csv_path: str, has_header=False):
    """Yield (p1, p2, p1_won) for each row in the CSV."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        if has_header:
            next(rdr, None)
        for row in rdr:
            if not row or len(row) < 4:
                continue
            # Expected columns:
            # [0]=game_number, [1]=player1_id, [2]=player2_id, [3]=W/L for player1
            p1 = row[1].strip()
            p2 = row[2].strip()
            outcome = row[3].strip().upper()
            if outcome not in ('W','L'):
                continue
            yield p1, p2, (outcome == 'W')

def main():
    ap = argparse.ArgumentParser(description="TrueSkill (win/loss only) rater with approximate Elo display.")
    ap.add_argument("--csv", required=True, help="Path to input CSV.")
    ap.add_argument("--has-header", action="store_true", help="Set if the CSV has a header row.")
    ap.add_argument("--mu0", type=float, default=25.0, help="Initial mu (default 25).")
    ap.add_argument("--sigma0", type=float, default=25.0/3.0, help="Initial sigma (default 25/3).")
    ap.add_argument("--beta", type=float, default=25.0/6.0, help="Performance noise beta (default 25/6).")
    ap.add_argument("--tau", type=float, default=0.0, help="Skill drift per game (default 0.0).")
    ap.add_argument("--kcon", type=float, default=3.0, help="Conservative k for mu - k*sigma (default 3).")
    ap.add_argument("--show-elo", action="store_true", help="Also show approximate Elo.")
    ap.add_argument("--elo-anchor", type=float, default=1500.0, help="Elo value corresponding to mu_anchor.")
    ap.add_argument("--mu-anchor", type=float, default=25.0, help="Mu value anchoring the Elo mapping (default 25).")
    ap.add_argument("--out", type=str, default="", help="Optional path to write ratings CSV.")
    args = ap.parse_args()

    # Initialize models
    ts = TrueSkillWL(mu0=args.mu0, sigma0=args.sigma0, beta=args.beta, tau=args.tau)

    # Stream matches in file order (assumed chronological)
    for p1, p2, p1_won in read_matches(args.csv, has_header=args.has_header):
        ts.update_match(p1, p2, p1_won)

    # Collect results and normalize ratings
    mu_values = {pid: ts.mu[pid] for pid in ts.mu.keys()}
    cons_values = {pid: ts.displayed(pid, k=args.kcon) for pid in ts.mu.keys()}
    
    # Normalize both standard and conservative ratings to BT range
    normalized_mu = normalize_ratings(mu_values)
    normalized_cons = normalize_ratings(cons_values)
    
    rows = []
    for pid in ts.mu.keys():
        mu = ts.mu[pid]
        sigma = math.sqrt(ts.var[pid])
        cons = ts.displayed(pid, k=args.kcon)
        
        if args.show_elo:
            # Convert normalized ratings to ELO using BT formula
            elo_mu = convert_to_elo(normalized_mu[pid], elo_anchor=args.elo_anchor)
            elo_cons = convert_to_elo(normalized_cons[pid], elo_anchor=args.elo_anchor)
        else:
            elo_mu = ""
            elo_cons = ""
            
        rows.append((pid, mu, sigma, cons, elo_mu, elo_cons))

    # Sort by conservative rating descending
    rows.sort(key=lambda r: r[3], reverse=True)

    # Print table
    hdr = ["player_id", "mu", "sigma", f"conservative(mu-{args.kcon}σ)"]
    if args.show_elo:
        hdr += ["elo(mu≈)", f"elo_conservative(mu-{args.kcon}σ≈)"]
    print(",".join(hdr))
    for r in rows:
        if args.show_elo:
            print(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.2f},{r[5]:.2f}")
        else:
            print(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f}")

    # Optionally write out
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            out_hdr = hdr
            w.writerow(out_hdr)
            for r in rows:
                if args.show_elo:
                    w.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.2f}", f"{r[5]:.2f}"])
                else:
                    w.writerow([r[0], f"{r[1]:.6f}", f"{r[2]:.6f}", f"{r[3]:.6f}"])

if __name__ == "__main__":
    main()
