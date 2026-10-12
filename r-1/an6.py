import numpy as np
from field_core import *
from mc import s_with_me, MYW, simulate
np.set_printoptions(precision=3,suppress=True)
names="RPS"; MY=TABLES[MY_TABLE]; EDGES=[30.,34.,38.,41.]; BN=["<30","30-34","34-38","38-41","41+"]
def buck(x): return np.searchsorted(EDGES,x,side='right')
NOPP=199
rng=np.random.default_rng(101)
a,_=simulate(80000,"levelk",rng,conc=12,lk_center=[.35,.40,.20,.05])
b,_=simulate(60000,"levelk_eq",rng,conc=12,lk_center=[.42,.30,.15,.05,.08])
c,_=simulate(60000,"ch",rng,tau_mu=1.5,tau_sd=0.5)
S_OTH=np.vstack([a,b,c]); np.save("blend12.npy",S_OTH)
SS=s_with_me(S_OTH,[0,0,1])[:,2]*100; P5=np.bincount(buck(SS),minlength=5)/len(SS)
print("FINAL BLEND (conc=12): mean sS=%.2f sd=%.2f"%(SS.mean(),SS.std()))
print("  bucket P:", " ".join(f"{BN[i]} {100*P5[i]:5.2f}%" for i in range(5)))

def crowd_pts(rng,sS,model):
    if model=="anchored_high":
        aa=rng.beta(1.6,1.4,NOPP); mu=aa*46+(1-aa)*(sS+rng.normal(0,5,NOPP))
        mu=np.where(rng.random(NOPP)<0.10,rng.uniform(20,60,NOPP),mu)
    elif model=="mixture":
        u=rng.random(NOPP)
        mu=np.where(u<.30,rng.normal(46,5,NOPP),np.where(u<.50,rng.normal(40,5,NOPP),
           np.where(u<.75,rng.normal(35,4,NOPP),np.where(u<.90,rng.normal(33.3,2,NOPP),rng.uniform(25,60,NOPP)))))
    elif model=="informed": mu=sS+rng.normal(0,6,NOPP)
    bb=buck(mu); A=np.zeros((NOPP,5)); A[np.arange(NOPP),bb]=100.0
    sp=rng.random(NOPP)<0.30; nb=np.clip(bb+rng.choice([-1,1],NOPP),0,4)
    A[sp,bb[sp]]=60.0; A[sp,nb[sp]]+=40.0
    return A
# build f_b tables averaged over the 3 crowd models
GRID=np.arange(0,101); F=np.zeros((5,101)); cnt=np.zeros(5)
rng=np.random.default_rng(4)
for t in range(9000):
    sS=SS[rng.integers(len(SS))]; tb=buck(sS)
    m=["anchored_high","mixture","informed"][t%3]
    cc=np.sort(crowd_pts(rng,sS,m)[:,tb])
    lo=np.searchsorted(cc,GRID,'left'); hi=np.searchsorted(cc,GRID,'right')
    F[tb]+=100.0*((lo+hi)/2)/NOPP; cnt[tb]+=1
p=cnt/cnt.sum(); F/=np.maximum(cnt,1)[:,None]
U=p[:,None]*F
# DP forcing EXACT sum 100
dp=np.full((6,101),-1e18); dp[0,0]=0.; ch=np.zeros((6,101),int)
for bi in range(5):
    for used in range(101):
        if dp[bi,used]<-1e17: continue
        for x in range(101-used):
            v=dp[bi,used]+U[bi,x]
            if v>dp[bi+1,used+x]: dp[bi+1,used+x]=v; ch[bi+1,used+x]=x
alloc=np.zeros(5,int); u=100
for bi in range(5,0,-1): alloc[bi-1]=ch[bi,u]; u-=ch[bi,u]
def rank_score(w): return float(sum(p[i]*F[i,int(round(w[i]))] for i in range(5)))
def abs_dev(w):    return float(200-2*np.dot(P5,w))
print("\n"+"="*84); print("GAME 3: RANKED vs ABSOLUTE - the two interpretations give OPPOSITE answers")
cands={
 "all-in 34-38 (modal)":       np.array([0,0,100,0,0]),
 "all-in 30-34 (runner-up)":   np.array([0,100,0,0,0]),
 "DP optimum (sum=100)":       alloc,
 "1/1/96/1/1":                 np.array([1,1,96,1,1]),
 "2/2/92/2/2":                 np.array([2,2,92,2,2]),
 "1/48/48/2/1":                np.array([1,48,48,2,1]),
 "5/30/30/20/15 (soft)":       np.array([5,30,30,20,15]),
 "uniform 20 each":            np.array([20,20,20,20,20]),
}
print(f"  {'allocation':30s} {'E[rank 0-100]':>14s} {'E[abs deviation]':>17s} {'abs as 100-dev/2':>18s}")
for k,w in cands.items():
    print(f"  {k:30s} {rank_score(w):14.2f} {abs_dev(w):17.2f} {100-abs_dev(w)/2:18.2f}")
print("\n  => 1/1/96/1/1 costs only %.2f deviation vs all-in but gains %.1f ranked points."
      %(abs_dev(np.array([1,1,96,1,1]))-abs_dev(np.array([0,0,100,0,0])),
        rank_score(np.array([1,1,96,1,1]))-rank_score(np.array([0,0,100,0,0]))))

print("\n"+"="*84); print("GAME 1 final (blend conc=12), P(each corner is best) and flip conditions")
g1=[]
for m in range(3):
    w=np.zeros(3); w[m]=1; sm=s_with_me(S_OTH,w); v=MY*multipliers(sm)
    g1.append(100*v[:,m]/v.max(axis=1))
sm=s_with_me(S_OTH,[0,0,1]); v=MY*multipliers(sm); am=v.argmax(axis=1)
print("  E[score]: R=%.2f P=%.2f S=%.2f | P(best corner) R=%.2f%% P=%.2f%% S=%.2f%%"
      %(g1[0].mean(),g1[1].mean(),g1[2].mean(),100*np.mean(am==0),100*np.mean(am==1),100*np.mean(am==2)))
print("  all-in S: p5=%.1f p25=%.1f median=%.1f | all-in P: p5=%.1f p25=%.1f median=%.1f"
      %(np.percentile(g1[2],5),np.percentile(g1[2],25),np.median(g1[2]),
        np.percentile(g1[1],5),np.percentile(g1[1],25),np.median(g1[1])))

print("\n"+"="*84); print("SKIP ANALYSIS: expected 0-100 score per question")
E={"Game1 (all-in S)":g1[2].mean(),"Game2 (28/40)":63.7,
   "Game3 ranked (1/1/96/1/1)":rank_score(np.array([1,1,96,1,1])),
   "Game3 absolute (100-dev/2)":100-abs_dev(np.array([0,0,100,0,0]))/2,
   "Game4 (B,B,B,S)":92.1}
for k,v in E.items(): print(f"   {k:32s} {v:6.2f}")
for g3key in ["Game3 ranked (1/1/96/1/1)","Game3 absolute (100-dev/2)"]:
    vals={"G1":E["Game1 (all-in S)"],"G2":E["Game2 (28/40)"],"G3":E[g3key],"G4":E["Game4 (B,B,B,S)"]}
    print(f"\n   -- if Game 3 is scored {g3key.split('(')[0].strip().split()[-1]}:")
    print(f"      answer all four -> mean {np.mean(list(vals.values())):.2f}")
    for q in vals:
        rest=[vals[x] for x in vals if x!=q]
        print(f"      SKIP {q} (excluded from mean) -> mean of rest {np.mean(rest):6.2f}   "
              +("BETTER" if np.mean(rest)>np.mean(list(vals.values())) else "worse"))
    print(f"      (if a skip scores 0 instead, never skip: every expected score above is >0)")

print("\n"+"="*84); print("DAY 2 LOOK-AHEAD: if today's realised shares are published, the natural Day-2")
print("  answer is the best response to them. Where does that land?")
for lab,s in [("today ~ blend median",np.array([0.323,0.327,0.350])),
              ("today = 33/33/34",np.array([1/3,1/3,1/3])),
              ("today = 30/32/38",np.array([0.30,0.32,0.38]))]:
    s=s/s.sum(); br,mv,_=best_response_profile(s)
    print(f"   {lab:22s} {np.round(s*100,1)} -> Day-2 L1 profile {np.round(br*100,1)}  moves {''.join(names[m] for m in mv)}")
print("   NOTE: a near-uniform Day-1 outcome maps to the L0 profile (20/34/46) as the Day-2 focal")
print("   answer, i.e. sS jumps toward 46. Day 2 is a DIFFERENT bet from Day 1.")
