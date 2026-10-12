import numpy as np, itertools
from field_core import *
from mc import s_with_me, MYW, simulate
np.set_printoptions(precision=3, suppress=True)
names="RPS"; MY=TABLES[MY_TABLE]
EDGES=[30.,34.,38.,41.]; BN=["<30","30-34","34-38","38-41","41+"]
def buck(x): return np.searchsorted(EDGES,x,side='right')
D=np.load("mc_out.npz")
SS=s_with_me(D["BLEND"],[0,0,1])[:,2]*100
P=np.bincount(buck(SS),minlength=5)/len(SS)
print("BLEND bucket probs:", dict(zip(BN,np.round(100*P,2))))

print("\n"+"="*80); print("GAME 3 UNDER *RANKED* SCORING (deviation ranked within college)")
print("  crowd model: each rival forms a belief mu, then 70% all-in on the bucket holding mu,")
print("  30% spread 60/40 over that bucket and its neighbour. (MY ASSUMPTION.)")
NOPP=199
def crowd_alloc(rng,sS,model):
    if model=="anchored_high":
        a=rng.beta(1.6,1.4,NOPP); mu=a*46+(1-a)*(sS+rng.normal(0,5,NOPP))
        mu=np.where(rng.random(NOPP)<0.10, rng.uniform(20,60,NOPP), mu)
    elif model=="mixture":
        u=rng.random(NOPP)
        mu=np.where(u<.30,rng.normal(46,5,NOPP),np.where(u<.50,rng.normal(40,5,NOPP),
           np.where(u<.75,rng.normal(35,4,NOPP),np.where(u<.90,rng.normal(33.3,2,NOPP),rng.uniform(25,60,NOPP)))))
    elif model=="informed": mu=sS+rng.normal(0,6,NOPP)
    b=buck(mu); A=np.zeros((NOPP,5)); A[np.arange(NOPP),b]=100.0
    sp=rng.random(NOPP)<0.30
    nb=np.clip(b+rng.choice([-1,1],NOPP),0,4)
    A[sp,b[sp]]=60.0; A[sp,nb[sp]]+=40.0
    return A
MYALLOC={f"all-in {BN[i]}":np.eye(5)[i]*100 for i in range(5)}
MYALLOC["50/50 30-34 & 34-38"]=np.array([0,50,50,0,0.])
MYALLOC["40/40/20 <30,30-34,34-38"]=np.array([40,40,20,0,0.])
MYALLOC["34/33/33 across 30-34,34-38,41+"]=np.array([0,34,33,0,33.])
for model in ["anchored_high","mixture","informed"]:
    rng=np.random.default_rng(4); acc={k:0.0 for k in MYALLOC}; T=4000
    crowdbucket=np.zeros(5)
    for t in range(T):
        sS=SS[rng.integers(len(SS))]; tb=buck(sS)
        A=crowd_alloc(rng,sS,model); crowdbucket+=A.sum(axis=0)/A.sum()
        dev_o=200-2*A[:,tb]; so=np.sort(dev_o)
        for k,w in MYALLOC.items():
            d=200-2*w[tb]
            lo=np.searchsorted(so,d,'left'); hi=np.searchsorted(so,d,'right')
            acc[k]+=100.0*(NOPP-(lo+hi)/2)/NOPP    # lower deviation = better rank
    print(f"\n  -- crowd={model};  crowd's own point mass by bucket: "
          +" ".join(f"{BN[i]} {100*crowdbucket[i]/T:4.1f}%" for i in range(5)))
    for k,v in sorted(acc.items(), key=lambda x:-x[1]): print(f"      {k:34s} E[rank score] {v/T:6.2f}")

print("\n"+"="*80); print("HOW WIDE IS 'WIDE'?  bucket probs vs Dirichlet concentration (scenario A centre)")
rng=np.random.default_rng(77)
for conc in [3,6,10,20,40,100]:
    s,_=simulate(120000,"levelk",rng,conc=conc,lk_center=[.35,.40,.20,.05])
    x=s_with_me(s,[0,0,1])[:,2]*100; p=np.bincount(buck(x),minlength=5)/len(x)
    print(f"  conc={conc:4d} sd(sS)={x.std():5.2f}  "+" ".join(f"{BN[i]} {100*p[i]:5.2f}%" for i in range(5))+f"   modal={BN[p.argmax()]}")

print("\n"+"="*80); print("TAIL RISK: a COORDINATED FOCAL BLOC (one widely-copied LLM answer)")
print("  With prob q, a bloc of fraction f all plays the SAME single move regardless of table.")
LK,_=level_profiles(6)
base=np.array([.35,.40,.20,.05])@LK[:4]
for mv in range(3):
    for f in [0.15,0.30,0.50]:
        s=(1-f)*base+f*np.eye(3)[mv]; s=0.85*s+0.15/3; s=(1-MYW)*s+MYW*np.array([0,0,1])
        v=MY*multipliers(s)
        print(f"   bloc plays {names[mv]}, f={f:.2f}: sR/sP/sS={np.round(s*100,1)} bucket={BN[buck(s[2]*100)]:6s} my best corner={names[v.argmax()]}")

print("\n"+"="*80); print("THREAT SCENARIO FOR GAME 1: an L1-HEAVY field (one BR round is the natural LLM answer)")
for c,lab in [([.25,.60,.12,.03],"L1-heavy .25/.60/.12/.03"),
              ([.30,.50,.15,.05],"L1-lean  .30/.50/.15/.05"),
              ([.35,.40,.20,.05],"your centre"),
              ([.42,.30,.15,.05],"my centre (no EQ type)")]:
    rng=np.random.default_rng(5)
    s,_=simulate(80000,"levelk",rng,conc=6.0,lk_center=c)
    r=[]
    for m in range(3):
        w=np.zeros(3); w[m]=1; sm=s_with_me(s,w); v=MY*multipliers(sm)
        r.append((100*v[:,m]/v.max(axis=1)).mean())
    x=s_with_me(s,[0,0,1])[:,2]*100; p=np.bincount(buck(x),minlength=5)/len(x)
    det=np.array(c)@LK[:4]; det=0.85*det+0.15/3
    print(f"  {lab:26s} det sR/sP/sS={np.round(det*100,1)}  E[G1] R={r[0]:5.1f} P={r[1]:5.1f} S={r[2]:5.1f}  best={names[int(np.argmax(r))]}  modal bucket={BN[p.argmax()]}")
