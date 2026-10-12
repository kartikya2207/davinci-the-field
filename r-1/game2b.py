import numpy as np
from mc import s_with_me
D=np.load("mc_out.npz"); SS_POOL=s_with_me(D["BLEND"],[0,0,1])[:,2]*100
NOPP=199; SPC=np.array([10,15,20,25,30.]); SPP=np.array([.25,.20,.30,.10,.15])
def crowd(rng,sS,model):
    if model=="anchored_high":
        a=rng.beta(1.6,1.4,NOPP); mu=a*46+(1-a)*(sS+rng.normal(0,5,NOPP))
        d=rng.random(NOPP)<0.10; mu=np.where(d,rng.uniform(20,60,NOPP),mu)
    elif model=="informed": mu=sS+rng.normal(0,6,NOPP)
    elif model=="mixture":
        u=rng.random(NOPP)
        mu=np.where(u<.30,rng.normal(46,5,NOPP),np.where(u<.50,rng.normal(40,5,NOPP),
           np.where(u<.75,rng.normal(35,4,NOPP),np.where(u<.90,rng.normal(33.3,2,NOPP),rng.uniform(25,60,NOPP)))))
    sp=rng.choice(SPC,NOPP,p=SPP); return mu-sp/2,mu+sp/2

print("CHECK: shape of the crowd's own P&L distribution (zero-sum by construction)")
for model in ["anchored_high","mixture","informed"]:
    rng=np.random.default_rng(9); allp=[]
    for t in range(400):
        sS=SS_POOL[rng.integers(len(SS_POOL))]; b,a=crowd(rng,sS,model)
        sa=np.sort(a); sb=np.sort(b); ca=np.r_[0,np.cumsum(sa)]; cb=np.r_[0,np.cumsum(sb)]
        nb=np.searchsorted(sa,b,'right'); ns=NOPP-np.searchsorted(sb,a,'left')
        opp=(nb*sS-(nb*b+ca[nb])/2)+((ns*a+(cb[NOPP]-cb[NOPP-ns]))/2-ns*sS); allp.append(opp)
    allp=np.concatenate(allp)
    print(f"  {model:15s} mean {allp.mean():8.3f}  MEDIAN {np.median(allp):8.3f}  skew {float(((allp-allp.mean())**3).mean()/allp.std()**3):7.2f}"
          f"  P(pnl>0)={100*np.mean(allp>0):5.1f}%  P(pnl==0)={100*np.mean(allp==0):5.1f}%")
print("  -> median > 0 with strong negative skew: a few large losers fund many small winners.")
print("     Quoting so wide that you never trade lands you BELOW the median. You must trade.\n")

# ---------- robustness: one quote scored across all 6 (crowd model x fill convention) cells
def run(model,my_at_mid,bids,spreads,ntrials=6000,seed=5):
    rng=np.random.default_rng(seed)
    B=np.repeat(bids,len(spreads)); SP=np.tile(spreads,len(bids)); A=B+SP
    sc=np.zeros(len(B)); idx=rng.integers(0,len(SS_POOL),ntrials)
    for t in range(ntrials):
        sS=SS_POOL[idx[t]]; b,a=crowd(rng,sS,model)
        sa=np.sort(a); sb=np.sort(b); ca=np.r_[0,np.cumsum(sa)]; cb=np.r_[0,np.cumsum(sb)]
        nb=np.searchsorted(sa,b,'right'); ns=NOPP-np.searchsorted(sb,a,'left')
        opp=(nb*sS-(nb*b+ca[nb])/2)+((ns*a+(cb[NOPP]-cb[NOPP-ns]))/2-ns*sS)
        mnb=np.searchsorted(sa,B,'right'); mns=NOPP-np.searchsorted(sb,A,'left')
        mine=((mnb*sS-(mnb*B+ca[mnb])/2)+((mns*A+(cb[NOPP]-cb[NOPP-mns]))/2-mns*sS)) if my_at_mid \
             else (mnb*(sS-B)+mns*(A-sS))
        so=np.sort(opp); lo=np.searchsorted(so,mine,'left'); hi=np.searchsorted(so,mine,'right')
        sc+=100.0*((lo+hi)/2)/NOPP
    return B,SP,A,sc/ntrials

bids=np.arange(24,41,1.0); spreads=np.arange(10,31,1.0)
cells={}
for model in ["anchored_high","mixture","informed"]:
    for mid in [False,True]:
        B,SP,A,sc=run(model,mid,bids,spreads); cells[(model,mid)]=sc
M=np.vstack([cells[k] for k in cells])
avg=M.mean(axis=0); worst=M.min(axis=0)
print("="*82); print("GAME 2 ROBUST CHOICE  (averaged over 3 crowd models x 2 fill conventions)")
print("   bid offer spr |  avg   worst |  anchHi/quote anchHi/mid  mix/quote  mix/mid  inf/quote inf/mid")
order=np.argsort(avg)[::-1]
for i in order[:10]:
    r=[cells[k][i] for k in cells]
    print(f"  {B[i]:4.0f} {A[i]:5.0f} {SP[i]:4.0f} | {avg[i]:5.2f}  {worst[i]:5.2f} |"+"".join(f"{x:11.2f}" for x in r))
print("\n  best by MAXIMIN (protect the worst crowd model):")
for i in np.argsort(worst)[::-1][:5]:
    print(f"  {B[i]:4.0f} {A[i]:5.0f} {SP[i]:4.0f} | avg {avg[i]:5.2f}  worst {worst[i]:5.2f}")
print("\n  how much does spread width matter (best quote at each spread, averaged)?")
for sp in range(10,31,2):
    m=SP==sp; j=np.argmax(np.where(m,avg,-1))
    print(f"    spread {sp:2d}: best bid {B[j]:.0f}/offer {A[j]:.0f}  avg {avg[j]:5.2f}  worst {worst[j]:5.2f}")
