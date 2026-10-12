"""BACKTEST: fit on Day 1 ONLY, predict Day 2 with the same machinery used for Day 3.
Day-1-only info: aggregate D1, N=467 -> M2 fit L0=0.694 L1=0.275 L2=0.031.
Unknown a priori: recruitment rate rho, inertia, llm size, l2. Use the SAME priors as mc3.
On Day 2 naive-L1 == recompute-L1 (BR(D1)==argmax payout), so the r-parameter is moot."""
import numpy as np
from core import *
rng=np.random.default_rng(0)
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs]); N=518
tabidx=np.resize(np.repeat(np.arange(6),np.round(wts*N).astype(int)),N)
NAIVE=np.array([int(np.argmax(t)) for t in tabs]); L2=np.array([int(np.argmax(values(t,L1))) for t in tabs])
T=40000; out=np.zeros((T,3))
for i in range(T):
    rho=rng.uniform(0.30,0.65); inert=rng.uniform(0,0.30); l2=rng.uniform(0,0.08); mom=rng.uniform(0,0.10)
    L0s=0.694*(1-rho); u=rng.random(N)
    isL2=u<l2; isL0=(u>=l2)&(u<l2+L0s); isL1=~isL2&~isL0
    mv=np.empty(N,int); n0=isL0.sum(); z=rng.random(n0); m0=rng.integers(0,3,n0)
    m0=np.where(z<inert,rng.choice(3,n0,p=D1/D1.sum()),m0); m0=np.where((z>=inert)&(z<inert+mom),2,m0)
    mv[isL0]=m0; mv[isL1]=NAIVE[tabidx[isL1]]; mv[isL2]=L2[tabidx[isL2]]
    out[i]=np.bincount(mv,minlength=3)/N
S=out[:,2]*100; R=out[:,0]*100; P=out[:,1]*100
print("BACKTEST Day-1-only -> Day-2 prediction:")
print("   R %.1f (actual 25.7)  P %.1f (actual 32.9)  S %.1f median %.1f (actual 41.3)"%(R.mean(),P.mean(),S.mean(),np.median(S)))
print("   S 5/25/75/95 pct: %s   P(S>=41)=%.1f%%   P(S in 38-41)=%.1f%%"%(np.round(np.percentile(S,[5,25,75,95]),1),100*(S>=41).mean(),100*((S>=38)&(S<41)).mean()))
print("   miss = %.1f points (r-2 missed by %.1f)"%(41.3-S.mean(),41.3-38.4))
# what would the Q3 allocator have entered for Day 2, and what would it have scored?
bk=np.bincount(bucket(S),minlength=5)/T; print("   Day-2 bucket forecast from Day-1-only: %s"%np.round(bk*100,1))
print("   -> the allocator would have put its mass on %s; actual bucket = >=41."%["<30","30-34","34-38","38-41",">=41"][bk.argmax()])
np.save("backtest_day2.npy",out)
