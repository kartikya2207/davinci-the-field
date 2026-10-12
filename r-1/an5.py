import numpy as np
from field_core import *
from mc import s_with_me, MYW, simulate
np.set_printoptions(precision=3, suppress=True)
names="RPS"; MY=TABLES[MY_TABLE]
EDGES=[30.,34.,38.,41.]; BN=["<30","30-34","34-38","38-41","41+"]
def buck(x): return np.searchsorted(EDGES,x,side='right')
NOPP=199

print("="*82); print("REBUILD THE BLEND AT SEVERAL WIDTHS (my conc=6 may have double-counted uncertainty:")
print("  the 3-model mixture ALREADY supplies model uncertainty)")
BL={}
for conc in [6,12,20]:
    rng=np.random.default_rng(101)
    a,_=simulate(80000,"levelk",rng,conc=conc,lk_center=[.35,.40,.20,.05])
    b,_=simulate(60000,"levelk_eq",rng,conc=conc,lk_center=[.42,.30,.15,.05,.08])
    c,_=simulate(60000,"ch",rng,tau_mu=1.5,tau_sd=0.5)
    s=np.vstack([a,b,c]); BL[conc]=s
    x=s_with_me(s,[0,0,1])[:,2]*100; p=np.bincount(buck(x),minlength=5)/len(x)
    o=np.argsort(p)[::-1]
    g1=[]
    for m in range(3):
        w=np.zeros(3); w[m]=1; sm=s_with_me(s,w); v=MY*multipliers(sm)
        g1.append((100*v[:,m]/v.max(axis=1)).mean())
    print(f"  conc={conc:3d}: mean sS={x.mean():5.2f} sd={x.std():5.2f} | "+" ".join(f"{BN[i]} {100*p[i]:5.2f}%" for i in range(5)))
    print(f"           modal={BN[o[0]]} over {BN[o[1]]} by {100*(p[o[0]]-p[o[1]]):.2f}pts | Game1 E[score] R={g1[0]:.1f} P={g1[1]:.1f} S={g1[2]:.1f} -> {names[int(np.argmax(g1))]}")

SS=s_with_me(BL[12],[0,0,1])[:,2]*100
P5=np.bincount(buck(SS),minlength=5)/len(SS)

print("\n"+"="*82); print("GAME 3: EXACT optimisation under RANKED scoring (separable DP, integer points)")
print("  E[rank] = sum_b P(true=b) * f_b(my points in b);  f_b nondecreasing step fn -> exact DP.")
def crowd_pts(rng,sS,model):
    if model=="anchored_high":
        a=rng.beta(1.6,1.4,NOPP); mu=a*46+(1-a)*(sS+rng.normal(0,5,NOPP))
        mu=np.where(rng.random(NOPP)<0.10,rng.uniform(20,60,NOPP),mu)
    elif model=="mixture":
        u=rng.random(NOPP)
        mu=np.where(u<.30,rng.normal(46,5,NOPP),np.where(u<.50,rng.normal(40,5,NOPP),
           np.where(u<.75,rng.normal(35,4,NOPP),np.where(u<.90,rng.normal(33.3,2,NOPP),rng.uniform(25,60,NOPP)))))
    elif model=="informed": mu=sS+rng.normal(0,6,NOPP)
    elif model=="allin_mode": mu=np.full(NOPP,35.5)+rng.normal(0,1.0,NOPP)   # crowd copies ONE answer
    b=buck(mu); A=np.zeros((NOPP,5)); A[np.arange(NOPP),b]=100.0
    sp=rng.random(NOPP)<0.30; nb=np.clip(b+rng.choice([-1,1],NOPP),0,4)
    A[sp,b[sp]]=60.0; A[sp,nb[sp]]+=40.0
    return A
GRID=np.arange(0,101)
for model in ["anchored_high","mixture","informed","allin_mode"]:
    rng=np.random.default_rng(4); T=6000
    F=np.zeros((5,101)); cnt=np.zeros(5)
    for t in range(T):
        sS=SS[rng.integers(len(SS))]; tb=buck(sS)
        c=np.sort(crowd_pts(rng,sS,model)[:,tb])
        lo=np.searchsorted(c,GRID,'left'); hi=np.searchsorted(c,GRID,'right')
        F[tb]+=100.0*((lo+hi)/2)/NOPP; cnt[tb]+=1
    p=cnt/T; F=np.divide(F,np.maximum(cnt,1)[:,None])
    U=p[:,None]*F                                    # utility of giving b exactly x points
    dp=np.full((6,101),-1e18); dp[0,0]=0.0; ch=np.zeros((6,101),int)
    for b in range(5):
        for used in range(101):
            if dp[b,used]<-1e17: continue
            for x in range(101-used):
                val=dp[b,used]+U[b,x]
                if val>dp[b+1,used+x]: dp[b+1,used+x]=val; ch[b+1,used+x]=x
    bestused=int(np.argmax(dp[5])); alloc=np.zeros(5,int); u=bestused
    for b in range(5,0,-1): alloc[b-1]=ch[b,u]; u-=ch[b,u]
    allin=np.array([ (p*F[np.arange(5),0]).sum()-p[i]*F[i,0]+p[i]*F[i,100] for i in range(5)])
    print(f"\n  crowd={model}   P(true bucket)="+" ".join(f"{BN[i]} {100*p[i]:4.1f}%" for i in range(5)))
    print(f"    OPTIMAL ALLOCATION {dict(zip(BN,alloc))}  E[rank score]={dp[5,bestused]:.2f}")
    print(f"    best all-in = {BN[int(np.argmax(allin))]} at E[rank score]={allin.max():.2f}   (gain from spreading {dp[5,bestused]-allin.max():+.2f})")

print("\n"+"="*82); print("GAME 3 UNDER *ABSOLUTE* (non-ranked) SCORING - the linear/improper case")
print("  E[score]=200-2*sum_b a_b p_b  -> all-in on the modal bucket. Probabilities (conc=12 blend):")
print("   "+" ".join(f"{BN[i]} {100*P5[i]:5.2f}%" for i in range(5)))
o=np.argsort(P5)[::-1]
print(f"   modal={BN[o[0]]} (E score {200-200*P5[o[0]]:.1f}), runner-up {BN[o[1]]} (E score {200-200*P5[o[1]]:.1f}), gap {100*(P5[o[0]]-P5[o[1]]):.2f}pts")
