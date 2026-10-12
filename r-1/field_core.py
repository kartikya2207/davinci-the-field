"""
The Field - Day 1. Core model.
All shares are FRACTIONS (0-1) internally. Percentages only at the reporting boundary.
"""
import numpy as np

# ---------------------------------------------------------------
# Tables: (pR, pP, pS) and population fraction. Index 0 is MY table.
# ---------------------------------------------------------------
TABLES = np.array([
    [3, 5, 9],   # 0  MINE
    [5, 9, 3],   # 1
    [9, 3, 5],   # 2
    [5, 3, 9],   # 3
    [3, 9, 5],   # 4
    [9, 5, 3],   # 5
], dtype=float)
TFRAC = np.array([0.34, 0.24, 0.14, 0.12, 0.10, 0.06])
MY_TABLE = 0
SMOOTH = 0.25
R, P, S = 0, 1, 2

assert abs(TFRAC.sum() - 1.0) < 1e-12, "table fractions must sum to 1"
assert np.all(TABLES[MY_TABLE] == [3, 5, 9]), "my table must be (R3,P5,S9)"


def multipliers(s):
    """m_m = share_of_move_that_m_beats / (share_of_m + 0.25).
    R beats S, P beats R, S beats P."""
    s = np.asarray(s, dtype=float)
    sR, sP, sS = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([sS / (sR + SMOOTH),
                     sR / (sP + SMOOTH),
                     sP / (sS + SMOOTH)], axis=-1)


def values(payoffs, s):
    """value vector for a player with `payoffs` facing aggregate shares `s`."""
    return np.asarray(payoffs, dtype=float) * multipliers(s)


def best_response_profile(s, tie_eps=0.0):
    """Aggregate share vector produced if EVERY table plays its argmax value move.
    Returns (shares, per_table_move, ties_found)."""
    v = values(TABLES, s)                       # (6,3)
    mv = v.max(axis=1)
    moves = v.argmax(axis=1)
    ties = int(np.sum(np.sum(v >= mv[:, None] - 1e-12, axis=1) > 1))
    out = np.zeros(3)
    for t in range(6):
        out[moves[t]] += TFRAC[t]
    return out, moves, ties


# ---------------------------------------------------------------
# Level-k profiles (each level best-responds to the level below, pure)
# ---------------------------------------------------------------
def level_profiles(max_level=6):
    """profiles[k] = aggregate share vector if the WHOLE field were level k.
    moves[k][t] = move played by a level-k player holding table t."""
    profiles = []
    moves = []
    # L0: play own highest payoff
    m0 = TABLES.argmax(axis=1)
    p0 = np.zeros(3)
    for t in range(6):
        p0[m0[t]] += TFRAC[t]
    profiles.append(p0); moves.append(m0)
    for k in range(1, max_level + 1):
        p, m, _ = best_response_profile(profiles[-1])
        profiles.append(p); moves.append(m)
    return np.array(profiles), np.array(moves)


# ---------------------------------------------------------------
# Cognitive Hierarchy: level k best-responds to the NORMALISED mixture
# of levels 0..k-1 under Poisson(tau), using CH's own lower-level behaviour.
# ---------------------------------------------------------------
def poisson_weights(tau, kmax):
    k = np.arange(kmax + 1)
    from math import lgamma
    w = np.exp(-tau + k * np.log(max(tau, 1e-12)) - np.array([lgamma(i + 1) for i in k]))
    return w


def ch_profiles(tau, kmax=6):
    """Returns profiles[k], moves[k] under Poisson-CH with precision tau."""
    pw = poisson_weights(tau, kmax)
    m0 = TABLES.argmax(axis=1)
    p0 = np.zeros(3)
    for t in range(6):
        p0[m0[t]] += TFRAC[t]
    profiles = [p0]; moves = [m0]
    for k in range(1, kmax + 1):
        w = pw[:k].copy()
        if w.sum() <= 0:
            w = np.ones(k)
        w = w / w.sum()
        belief = (w[:, None] * np.array(profiles[:k])).sum(axis=0)
        p, m, _ = best_response_profile(belief)
        profiles.append(p); moves.append(m)
    return np.array(profiles), np.array(moves), pw


def ch_aggregate(tau, kmax=8):
    """Actual population aggregate under CH: sum_k Poisson(k;tau) * profile_k,
    with the Poisson tail beyond kmax folded into level kmax."""
    profiles, moves, _ = ch_profiles(tau, kmax)
    w = poisson_weights(tau, kmax)
    tail = max(0.0, 1.0 - w.sum())
    w = w.copy(); w[-1] += tail
    return (w[:, None] * profiles).sum(axis=0), profiles, moves, w


# ---------------------------------------------------------------
# Game 1 score
# ---------------------------------------------------------------
def game1_score(weights, payoffs, s):
    v = values(payoffs, s)
    return 100.0 * float(np.dot(weights, v) / v.max())
