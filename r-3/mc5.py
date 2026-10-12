"""MC v3 = mc4 but the LLM bloc is split into 3 vendor sub-blocs (ChatGPT/Gemini/Claude users), each with its own
prompt-variant draw and near-tie draw. Less than perfect correlation -> tests tail sensitivity."""
import numpy as np, sys
from core import *
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0; T=int(sys.argv[2]) if len(sys.argv)>2 else 100000
rng=np.random.default_rng(seed)
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs])
tabidx=np.resize(np.repeat(np.arange(6),np.round(wts*N3).astype(int)),N3)
EXTRAP=np.array([0.21,0.33,0.45])
V1c=np.array([int(np.argmax(t)) for t in tabs]); V2c=np.array([int(np.argmax(values(t,D2))) for t in tabs]); V3c=np.array([int(np.argmax(values(t,EXTRAP))) for t in tabs])
L2c=np.array([int(np.argmax(values(t,np.array([.44,.10,.46])))) for t in tabs])
out=np.zeros((T,3))
for i in range(T):
    llm=rng.uniform(0.20,0.80); rho=rng.uniform(0.30,0.65); inert=rng.uniform(0,0.30)
    pR2=rng.uniform(0.45,0.85); r_hum=rng.beta(3,2); l2=rng.uniform(0,0.06); mom=rng.uniform(0,0.10)
    L0s=0.35*(1-rho); u=rng.random(N3)
    isL2=u<l2; isL0=(u>=l2)&(u<l2+L0s); isL1=~isL2&~isL0; isllm=rng.random(N3)<llm
    vendor=rng.choice(3,N3,p=[.55,.3,.15])
    mv=np.empty(N3,int)
    n0=isL0.sum(); z=rng.random(n0); m0=rng.integers(0,3,n0)
    m0=np.where(z<inert,rng.choice(3,n0,p=D2/D2.sum()),m0); m0=np.where((z>=inert)&(z<inert+mom),2,m0); mv[isL0]=m0
    choice=np.empty(N3,int)
    for vd in range(3):
        pv=rng.dirichlet([3.0,4.5,2.5]); var=rng.choice(3,N3,p=pv)
        v2=V2c.copy()
        if rng.random()>pR2: v2[1]=1
        c=np.where(var==0,V1c[tabidx],np.where(var==1,v2[tabidx],V3c[tabidx]))
        choice[vendor==vd]=c[vendor==vd]
    m=isL1&isllm; mv[m]=choice[m]
    hum=isL1&~isllm; hr=rng.random(N3)<r_hum; mv[hum]=np.where(hr,V2c[tabidx],V1c[tabidx])[hum]
    mv[isL2]=L2c[tabidx[isL2]]
    out[i]=np.bincount(mv,minlength=3)/N3
np.save("mc5_out_s%d.npy"%seed,out)
S=out[:,2]*100;R=out[:,0]*100;P=out[:,1]*100
print("mc5 seed %d | R %.1f sd %.1f | P %.1f sd %.1f | S %.1f med %.1f sd %.2f | pct 5/95 %s"%(seed,R.mean(),R.std(),P.mean(),P.std(),S.mean(),np.median(S),S.std(),np.round(np.percentile(S,[5,95]),1)))
print(" BUCKETS:",np.round(np.bincount(bucket(S),minlength=5)/T*100,2)," within±1 of 41: %.1f%%"%(100*((S>=40)&(S<42)).mean()))
V=np.array([values(MINE,s) for s in out]); rat=V/V.max(1,keepdims=True); np.save("mc5_V_s%d.npy"%seed,V)
print(" Q1 P(best R/P/S)=%s  E[v/vmax]=%s"%(np.round(np.bincount(V.argmax(1),minlength=3)/T*100,1),np.round(rat.mean(0)*100,2)))
for w in [(0,0,100),(0,30,70),(0,50,50),(0,100,0)]:
    sc=100*(rat@(np.array(w)/100)); print("   mix %s: E=%.2f p10=%.1f P(<70)=%.1f%%"%(w,sc.mean(),np.percentile(sc,10),100*(sc<70).mean()))
