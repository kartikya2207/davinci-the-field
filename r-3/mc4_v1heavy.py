"""Day-3 MC v2: LLM bloc split by PROMPT VARIANT the student pasted.
V1 = question only  -> argmax payout (R20/P34/S46 ladder)
V2 = + Day-2 aggregate -> BR to D2 ((5,9,3)->R by 2%: LLM says Rock w.p. pR2, else Paper/mix)
V3 = + both days, trend -> BR to extrapolated R21/P33/S45 ((5,9,3)->R, (5,3,9)->R)
Each prompt variant is ONE correlated draw per table (bloc moves together)."""
import numpy as np, sys
from core import *
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0; T=int(sys.argv[2]) if len(sys.argv)>2 else 100000
rng=np.random.default_rng(seed)
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs])
tabidx=np.resize(np.repeat(np.arange(6),np.round(wts*N3).astype(int)),N3)
EXTRAP=np.array([0.21,0.33,0.45])
V1c=np.array([int(np.argmax(t)) for t in tabs]); V2c=np.array([int(np.argmax(values(t,D2))) for t in tabs]); V3c=np.array([int(np.argmax(values(t,EXTRAP))) for t in tabs])
A_rec=np.array([0.44,0.10,0.46]); L2c=np.array([int(np.argmax(values(t,A_rec))) for t in tabs])
print("V1",V1c,"V2",V2c,"V3",V3c,"L2",L2c)
out=np.zeros((T,3)); par=np.zeros((T,8))
for i in range(T):
    llm=rng.uniform(0.20,0.80); rho=rng.uniform(0.30,0.65); inert=rng.uniform(0,0.30)
    pv=rng.dirichlet([6.0,3.0,1.0])            # (V1,V2,V3) shares among LLM users, mean (.3,.45,.25)
    pR2=rng.uniform(0.45,0.85)                 # P(LLM says Rock for (5,9,3) at the 2% near-tie)
    r_hum=rng.beta(3,2)                        # human sharp: P(recompute vs D2)
    l2=rng.uniform(0,0.08); mom=rng.uniform(0,0.10)
    L0s=0.35*(1-rho); u=rng.random(N3)
    isL2=u<l2; isL0=(u>=l2)&(u<l2+L0s); isL1=~isL2&~isL0; isllm=rng.random(N3)<llm
    mv=np.empty(N3,int)
    n0=isL0.sum(); z=rng.random(n0); m0=rng.integers(0,3,n0)
    m0=np.where(z<inert,rng.choice(3,n0,p=D2/D2.sum()),m0); m0=np.where((z>=inert)&(z<inert+mom),2,m0); mv[isL0]=m0
    # LLM users: assign prompt variant per user, advice per (variant,table) is deterministic except the V2 near-tie
    var=rng.choice(3,N3,p=pv)
    v2=V2c.copy()
    if rng.random()>pR2: v2[1]=1               # bloc-level draw: this time the LLM said Paper for (5,9,3)
    choice=np.where(var==0,V1c[tabidx],np.where(var==1,v2[tabidx],V3c[tabidx]))
    m=isL1&isllm; mv[m]=choice[m]
    hum=isL1&~isllm; hr=rng.random(N3)<r_hum
    mv[hum]=np.where(hr,V2c[tabidx],V1c[tabidx])[hum]
    mv[isL2]=L2c[tabidx[isL2]]
    out[i]=np.bincount(mv,minlength=3)/N3; par[i]=[llm,rho,inert,pv[0],pv[1],pv[2],pR2,l2]
np.save("mc4v1heavy_out_s%d.npy"%seed,out); np.save("mc4v1heavy_par_s%d.npy"%seed,par)
S=out[:,2]*100;R=out[:,0]*100;P=out[:,1]*100
print("seed %d T=%d | R %.1f sd %.1f | P %.1f sd %.1f | S mean %.1f med %.1f sd %.2f"%(seed,T,R.mean(),R.std(),P.mean(),P.std(),S.mean(),np.median(S),S.std()))
print(" S pct 2.5/5/25/50/75/95/97.5:",np.round(np.percentile(S,[2.5,5,25,50,75,95,97.5]),1))
print(" BUCKETS:",np.round(np.bincount(bucket(S),minlength=5)/T*100,2)," within±1 of 41: %.1f%%  of 38: %.1f%%"%(100*((S>=40)&(S<42)).mean(),100*((S>=37)&(S<39)).mean()))
V=np.array([values(MINE,s) for s in out]); rat=V/V.max(1,keepdims=True); np.save("mc4v1heavy_V_s%d.npy"%seed,V)
print(" Q1 P(best R/P/S)=%s  E[v/vmax]=%s"%(np.round(np.bincount(V.argmax(1),minlength=3)/T*100,1),np.round(rat.mean(0)*100,2)))
