"""Monte Carlo over the joint distribution of (sR,sP,sS)."""
import numpy as np
from field_core import *

N_TOT = 200          # players in the field, INCLUDING me
N_OTH = N_TOT - 1    # 199 others
MYW   = 1.0/N_TOT    # my weight in the aggregate = 0.5%
KMAX  = 8

# ---- profile libraries -------------------------------------------------
LK_PROFILES, LK_MOVES = level_profiles(KMAX)          # level-k, pure ladder
S_EQ = np.load("s_eq.npy")                            # exact mixed equilibrium aggregate

# CH profiles cached on a tau grid (CH profiles are piecewise-constant in tau)
TAU_GRID = np.round(np.arange(0.05, 6.0001, 0.0025), 6)
_ch_prof = np.empty((len(TAU_GRID), KMAX+1, 3))
_ch_q    = np.empty((len(TAU_GRID), KMAX+1))
for i, tau in enumerate(TAU_GRID):
    p, m, _ = ch_profiles(tau, KMAX)
    _ch_prof[i] = p
    w = poisson_weights(tau, KMAX); w = w/w.sum()      # truncate+renormalise at KMAX
    _ch_q[i] = w

def _draw_shares(pmove_pure, h, rng):
    """pmove_pure (C,3) pure-player move distribution; h (C,) hedger prob.
    Returns s_others (C,3) over the 199 non-me players."""
    C = len(h)
    nh = rng.binomial(N_OTH, h)                        # hedgers: contribute exactly (1/3,1/3,1/3)
    npure = N_OTH - nh
    cnt = rng.multinomial(npure, pmove_pure)           # vectorised over trials
    s = (cnt + (nh/3.0)[:, None]) / N_OTH
    return s

def simulate(n, scenario, rng, conc=6.0, hedge_mean=0.15, hedge_conc=12.0,
             lk_center=None, tau_mu=1.5, tau_sd=0.5):
    """Returns s_others (n,3) and a dict of drawn parameters."""
    h = rng.beta(hedge_mean*hedge_conc, (1-hedge_mean)*hedge_conc, n)
    info = {"hedge": h}
    if scenario == "levelk":
        c = np.array(lk_center, float); c = c/c.sum()
        K = len(c) - 1                                  # levels 0..K  (last entry may be 'EQ')
        q = rng.dirichlet(conc*c, n)
        info["q"] = q
        pmove = q @ LK_PROFILES[:len(c)]
    elif scenario == "levelk_eq":
        # last component of lk_center is the EQUILIBRIUM type, not a level
        c = np.array(lk_center, float); c = c/c.sum()
        q = rng.dirichlet(conc*c, n); info["q"] = q
        lib = np.vstack([LK_PROFILES[:len(c)-1], S_EQ[None, :]])
        pmove = q @ lib
    elif scenario == "ch":
        tau = np.clip(rng.normal(tau_mu, tau_sd, n), 0.05, 6.0)
        info["tau"] = tau
        idx = np.searchsorted(TAU_GRID, tau).clip(0, len(TAU_GRID)-1)
        pmove = np.einsum('nk,nkm->nm', _ch_q[idx], _ch_prof[idx])
    else:
        raise ValueError(scenario)
    pmove = np.clip(pmove, 0, None); pmove /= pmove.sum(axis=1, keepdims=True)
    return _draw_shares(pmove, h, rng), info


def s_with_me(s_oth, my_w):
    """Aggregate including my 100 throws with weight vector my_w."""
    return (1-MYW)*s_oth + MYW*np.asarray(my_w, float)
