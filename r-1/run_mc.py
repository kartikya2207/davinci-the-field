import numpy as np, json
from field_core import *
from mc import *
np.set_printoptions(precision=3, suppress=True)

N = 200_000
rng = np.random.default_rng(20260902)

# ---------------- scenario definitions (ASSUMPTIONS - flagged in the writeup) ----
SCEN = {
 # (A) exactly the user's stated prior, so results are comparable to the report
 "A_user_levelk":  dict(scenario="levelk",    lk_center=[.35,.40,.20,.05], conc=6.0),
 # (B) my prior: L0-heavier (L0 = the Bayes answer under uniform share beliefs)
 #     + an EQUILIBRIUM type for well-prompted players who find the cycle
 "B_mine_levelk":  dict(scenario="levelk_eq", lk_center=[.42,.30,.15,.05,.08], conc=6.0),
 # (C) Poisson-CH, tau ~ N(1.5,0.5)
 "C_ch":           dict(scenario="ch",        tau_mu=1.5, tau_sd=0.5),
}
BLEND_W = {"A_user_levelk":0.40, "B_mine_levelk":0.30, "C_ch":0.30}

res={}
for k,kw in SCEN.items():
    s,info = simulate(N, rng=rng, **kw)
    res[k]=s
    assert np.all(s>=-1e-12) and np.all(s<=1+1e-12), "share out of [0,1]!"
    assert np.allclose(s.sum(axis=1),1,atol=1e-9), "shares do not sum to 1!"
    print(f"{k:16s} mean sR/sP/sS = {s.mean(axis=0)*100} sd sS={s[:,2].std()*100:.2f}")

# blended ensemble
counts={k:int(round(N*w)) for k,w in BLEND_W.items()}
blend=np.vstack([res[k][:counts[k]] for k in SCEN])
rng.shuffle(blend)
res["BLEND"]=blend
print(f"{'BLEND':16s} mean sR/sP/sS = {blend.mean(axis=0)*100} sd sS={blend[:,2].std()*100:.2f}  N={len(blend)}")

np.savez_compressed("mc_out.npz", **res)
print("\nsaved mc_out.npz")

print("\n"+"="*72)
print("SANITY: hand-check the deterministic centres against the MC means")
print(" level-k mixture 35/40/20/5 with NO hedgers and NO sampling noise:")
lk,_=level_profiles(4)
det=np.array([.35,.40,.20,.05])@lk[:4]
print("   ->",np.round(det*100,2),"  (your hand figure: 33.4 / 31.9 / 34.7)")
print(" with 15% hedgers folded in: ", np.round((0.85*det+0.15/3)*100,2))
agg,_,_,_=ch_aggregate(1.5,kmax=8)
print(" CH tau=1.5 no hedgers:",np.round(agg*100,2),"  with 15% hedgers:",np.round((0.85*agg+0.15/3)*100,2))
