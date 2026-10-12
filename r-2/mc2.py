import numpy as np
from core import TABLES, values
from math import exp, factorial
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs]); N=467
A1=np.load("A1.npy")   # rows = aggregate if ALL play level k (ladder is anchor-invariant: BR(D1)==BR(unif))
D1=np.array([.307,.325,.369])

# ---- Poisson-CH one-parameter fit to Day 1 (stronger test than free mixture) ----
def ch(tau,K=4):
    f=np.array([exp(-tau)*tau**k/factorial(k) for k in range(K)]); return f/f.sum()
best=min(((np.abs((ch(t)@A1)-D1).sum(),t) for t in np.arange(0.05,4,0.005)))
tau1=best[1]; print("Poisson-CH tau1=%.3f err=%.4f -> %s  weights %s"%(tau1,best[0],np.round(ch(tau1)@A1*100,2),np.round(ch(tau1),3)))

# per-agent level-k choice table: choice[k][table_index] = move index
def kchoice(k,t):
    if k==0: return None
    bel=np.array([1/3,1/3,1/3]) if False else D1
    for i in range(1,k): bel=A1[i]
    return int(np.argmax(values(t,bel)))
CH_=np.array([[ (0 if k==0 else int(np.argmax(values(t, D1 if k==1 else A1[k-1])))) for t in tabs] for k in range(5)])

rng=np.random.default_rng(11); T=40000
tabidx=np.concatenate([[i]*int(round(w*N)) for i,w in enumerate(wts)])
tabidx=np.resize(tabidx,N)
out=np.zeros((T,3))
meta=np.zeros((T,3))
for tr in range(T):
    # --- sensitivity axes, drawn per trial ---
    llm   = rng.uniform(0.20,0.70)          # LLM-bloc size
    inert = rng.uniform(0.10,0.35)          # repeat Day-1 / default tail
    dtau  = rng.uniform(0.10,1.00)          # feedback-driven level climb (Nagel/Weber)
    p_llm_l2 = rng.uniform(0.10,0.40)       # frac of LLM bloc that reasons to L2
    tau2=tau1+dtau; f=ch(tau2,5)
    r=rng.random(N)
    mv=np.empty(N,int)
    # inertial tail: reproduce Day-1 marginal
    is_in = r<inert
    mv[is_in]=rng.choice(3,is_in.sum(),p=D1/D1.sum())
    rem=~is_in
    r2=rng.random(N)
    is_llm = rem & (r2 < llm)
    # LLM bloc: correlated low-level argmax mass, mostly L1, some L2, small rock-bias slip
    lv = np.where(rng.random(N)<p_llm_l2,2,1)
    mvl = CH_[lv,tabidx]
    rock_bias = rng.random(N) < 0.05        # documented LLM positional Rock bias
    mvl = np.where(rock_bias,0,mvl)
    mv[is_llm]=mvl[is_llm]
    hum = rem & ~is_llm
    lvh = rng.choice(5,N,p=f)
    mvh = np.where(lvh==0, rng.integers(0,3,N), CH_[np.clip(lvh,0,4),tabidx])
    mv[hum]=mvh[hum]
    out[tr]=np.bincount(mv,minlength=3)/N
    meta[tr]=[llm,inert,dtau]
np.save("mc_out.npy",out)
sS=out[:,2]*100; sR=out[:,0]*100; sP=out[:,1]*100
print("\nDAY-2 PREDICTIVE (T=%d)"%T)
print(" mean R/P/S = %.1f / %.1f / %.1f"%(sR.mean(),sP.mean(),sS.mean()))
print(" S: median %.1f  sd %.2f  5/25/75/95 %s"%(np.median(sS),sS.std(),np.round(np.percentile(sS,[5,25,75,95]),1)))
print(" R: median %.1f  sd %.2f  95th %.1f  P(R>46%%)=%.4f"%(np.median(sR),sR.std(),np.percentile(sR,95),(sR>46).mean()))
b=[(sS<30),(sS>=30)&(sS<34),(sS>=34)&(sS<38),(sS>=38)&(sS<41),(sS>=41)]
P=np.array([x.mean() for x in b])
print(" BUCKETS <30 /30-34 /34-38 /38-41 />=41 = %s"%np.round(P*100,2))
np.save("bucketP.npy",P)
