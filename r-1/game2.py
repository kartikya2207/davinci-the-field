import numpy as np
from mc import s_with_me
np.set_printoptions(precision=2, suppress=True)
D=np.load("mc_out.npz")
SS_POOL=s_with_me(D["BLEND"],[0,0,1])[:,2]*100
NOPP=199
SPC=np.array([10,15,20,25,30.]); SPP=np.array([.25,.20,.30,.10,.15])

def crowd(rng,sS,model):
    if model=="anchored_high":
        a=rng.beta(1.6,1.4,NOPP)
        mu=a*46+(1-a)*(sS+rng.normal(0,5,NOPP))
        d=rng.random(NOPP)<0.10; mu=np.where(d,rng.uniform(20,60,NOPP),mu)
    elif model=="informed":
        mu=sS+rng.normal(0,6,NOPP)
    elif model=="mixture":
        u=rng.random(NOPP)
        mu=np.where(u<.30,rng.normal(46,5,NOPP),np.where(u<.50,rng.normal(40,5,NOPP),
           np.where(u<.75,rng.normal(35,4,NOPP),np.where(u<.90,rng.normal(33.3,2,NOPP),rng.uniform(25,60,NOPP)))))
    sp=rng.choice(SPC,NOPP,p=SPP)
    return mu-sp/2, mu+sp/2

def run(model,my_at_mid,ntrials=8000,bids=np.arange(20,46,1.0),spreads=np.arange(10,31,1.0),seed=5):
    rng=np.random.default_rng(seed)
    B=np.repeat(bids,len(spreads)); SP=np.tile(spreads,len(bids)); A=B+SP
    sc=np.zeros(len(B)); pnl=np.zeros(len(B)); ntr=np.zeros(len(B)); win=np.zeros(len(B))
    zerosum=[]
    idx=rng.integers(0,len(SS_POOL),ntrials)
    for t in range(ntrials):
        sS=SS_POOL[idx[t]]
        b,a=crowd(rng,sS,model)
        sa=np.sort(a); sb=np.sort(b)
        ca=np.concatenate([[0],np.cumsum(sa)]); cb=np.concatenate([[0],np.cumsum(sb)])
        # crowd P&L: ALWAYS midpoint of the two crossing quotes -> exactly zero sum
        nb=np.searchsorted(sa,b,side='right'); ns=NOPP-np.searchsorted(sb,a,side='left')
        opp=(nb*sS-(nb*b+ca[nb])/2)+((ns*a+(cb[NOPP]-cb[NOPP-ns]))/2-ns*sS)
        zerosum.append(opp.sum())
        mnb=np.searchsorted(sa,B,side='right'); mns=NOPP-np.searchsorted(sb,A,side='left')
        if my_at_mid:
            mine=(mnb*sS-(mnb*B+ca[mnb])/2)+((mns*A+(cb[NOPP]-cb[NOPP-mns]))/2-mns*sS)
        else:   # I get filled at MY OWN quote (worse for me: I pay my bid / receive my offer)
            mine=mnb*(sS-B)+mns*(A-sS)
        so=np.sort(opp)
        lo=np.searchsorted(so,mine,side='left'); hi=np.searchsorted(so,mine,side='right')
        sc+=100.0*((lo+hi)/2)/NOPP; pnl+=mine; ntr+=mnb+mns; win+=(mine>0)
    return B,SP,A,sc/ntrials,pnl/ntrials,ntr/ntrials,win/ntrials,np.array(zerosum)

print("VALIDATION: crowd P&L must sum to exactly 0 each trial (midpoint rule)")
_,_,_,_,_,_,_,zs=run("mixture",True,ntrials=300)
print(f"  max |sum of crowd P&L| over 300 trials = {np.abs(zs).max():.3e}   -> zero-sum OK\n")

for model in ["anchored_high","mixture","informed"]:
    for mid in [False,True]:
        B,SP,A,sc,pnl,nt,win,_=run(model,mid)
        inwin=(B>=24)&(B<=40)
        o=np.argsort(sc)[::-1]; ow=o[inwin[o]]
        print("="*80); print(f"GAME 2  crowd={model}  my fill at {'MIDPOINT' if mid else 'MY QUOTE'}")
        print("  best within your bid grid 24-40:")
        print("        bid  offer  spread  E[rankscore]   E[P&L]  P(P&L>0)  E[#trades]")
        for i in ow[:6]:
            print(f"       {B[i]:5.1f} {A[i]:6.1f} {SP[i]:6.1f}    {sc[i]:8.2f}  {pnl[i]:8.1f}    {100*win[i]:5.1f}%    {nt[i]:6.1f}")
        i=o[0]
        print(f"  unconstrained best (bid 20-45): bid {B[i]:.0f} offer {A[i]:.0f} spread {SP[i]:.0f} -> score {sc[i]:.2f}, P&L {pnl[i]:.1f}, P(win) {100*win[i]:.1f}%")
        for sp in [10,20,30]:
            m=SP==sp; j=np.argmax(np.where(m,sc,-1))
            print(f"    best with spread={sp}: bid {B[j]:.0f}/offer {A[j]:.0f} score {sc[j]:.2f} P&L {pnl[j]:.1f} trades {nt[j]:.1f}")
