"""Day-3 agent-based MC. 518 agents. Types: L0(uniform), INERTIA(replay D2 marginal),
L1-naive (argmax payout), L1-recompute (BR to published D2), L2 (BR to anticipated
recompute aggregate), momentum L0 (focal on the visible winner, Scissors).
Correlated LLM bloc: all LLM users of the same table get the SAME advice draw."""
import numpy as np, sys
from core import *
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
T=int(sys.argv[2]) if len(sys.argv)>2 else 100000
rng=np.random.default_rng(seed)
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs])
tabidx=np.repeat(np.arange(6),np.round(wts*N3).astype(int)); tabidx=np.resize(tabidx,N3)
NAIVE=np.array([int(np.argmax(t)) for t in tabs])                     # argmax payout
RECOMP=np.array([int(np.argmax(values(t,D2))) for t in tabs])         # BR to published D2
A_rec=np.array([0.44,0.10,0.46])
L2=np.array([int(np.argmax(values(t,A_rec))) for t in tabs])          # BR to anticipated rotation
print("naive",NAIVE,"recomp",RECOMP,"L2",L2)
out=np.zeros((T,3)); par=np.zeros((T,6))
for i in range(T):
    # ---- sensitivity axes ----
    llm   = rng.uniform(0.20,0.80)      # LLM-user share of the field
    rho   = rng.uniform(0.30,0.65)      # fraction of remaining L0 recruited to L1 this round (day2 obs: ~0.49)
    inert = rng.uniform(0.00,0.30)      # share of non-L1 that replays D2 rather than uniform
    r_llm = rng.beta(2.4,2.0)           # P(an LLM answer uses published D2 -> recompute)  mean .55
    r_hum = rng.beta(3,2)               # same for human sharp players                     mean .6
    l2    = rng.uniform(0.00,0.08)      # level-2 share (days 1-2 fitted ~2-3%)
    mom   = rng.uniform(0.00,0.10)      # share of L0 that plays the visible winner (S)
    L0share = 0.35*(1-rho)              # day-2 L0 was 0.35 (M2 fit)
    L1share = 1-L0share-l2
    # ---- assign types ----
    u=rng.random(N3)
    isL2 = u<l2
    isL0 = (u>=l2)&(u<l2+L0share)
    isL1 = ~isL2 & ~isL0
    isllm = rng.random(N3)<llm
    mv=np.empty(N3,int)
    # L0: uniform / inertia / momentum
    n0=isL0.sum(); z=rng.random(n0)
    m0=rng.integers(0,3,n0)
    m0=np.where(z<inert, rng.choice(3,n0,p=D2/D2.sum()), m0)
    m0=np.where((z>=inert)&(z<inert+mom), 2, m0)
    mv[isL0]=m0
    # L1: correlated LLM advice per table; humans independent
    adv_llm = rng.random(6)<r_llm       # ONE draw per table: the bloc moves together
    llm_choice = np.where(adv_llm,RECOMP,NAIVE)
    # LLM tie-awareness on (5,9,3): 2% margin -> some fraction of recompute advice still says Paper/mix
    tie_paper = rng.random()<0.35
    if adv_llm[1] and tie_paper: llm_choice[1]=1
    hum_rec = rng.random(N3)<r_hum
    hum_choice = np.where(hum_rec, RECOMP[tabidx], NAIVE[tabidx])
    mv[isL1 & isllm] = llm_choice[tabidx[isL1 & isllm]]
    mv[isL1 & ~isllm] = hum_choice[isL1 & ~isllm]
    mv[isL2] = L2[tabidx[isL2]]
    out[i]=np.bincount(mv,minlength=3)/N3
    par[i]=[llm,rho,inert,r_llm,l2,mom]
np.save("mc3_out_s%d.npy"%seed,out); np.save("mc3_par_s%d.npy"%seed,par)
S=out[:,2]*100; R=out[:,0]*100; P=out[:,1]*100
print("seed %d T=%d"%(seed,T))
print(" R mean %.1f sd %.2f  | P mean %.1f sd %.2f | S mean %.1f median %.1f sd %.2f"%(R.mean(),R.std(),P.mean(),P.std(),S.mean(),np.median(S),S.std()))
print(" S pct 2.5/5/25/50/75/95/97.5 =",np.round(np.percentile(S,[2.5,5,25,50,75,95,97.5]),1))
bk=bucket(S); Pb=np.bincount(bk,minlength=5)/T
print(" BUCKETS <30/30-34/34-38/38-41/>=41 = %s"%np.round(Pb*100,2))
print(" mass within +/-1 of 41: %.1f%%   of 38: %.1f%%"%(100*((S>=40)&(S<42)).mean(),100*((S>=37)&(S<39)).mean()))
V=np.array([values(MINE,s) for s in out]); rat=V/V.max(1,keepdims=True)
print(" Q1 P(best=R/P/S)=%s  E[v/vmax] R/P/S=%s"%(np.round(np.bincount(V.argmax(1),minlength=3)/T*100,1),np.round(rat.mean(0)*100,2)))
np.save("mc3_V_s%d.npy"%seed,V)
