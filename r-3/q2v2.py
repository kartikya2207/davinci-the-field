"""Q2 v2: explicit bull/bear tails. Field = defaults + LLM-centred + wide humans + BULLS (bid>=42) + BEARS (offer<=42).
Sweep bull share 2-12%, bear share 0-6%. Rank-score every legal quote under 3 conventions. Report robust pick."""
import numpy as np, sys, pickle
from core import *
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
A=np.load(sys.argv[2] if len(sys.argv)>2 else "ensemble_out.npy")[:,2]*100
def field(n,rs,bull,bear,center):
    r=rs.random(n); B=np.empty(n); O=np.empty(n)
    c=np.cumsum([0.28,0.42,0.30-bull-bear,bull,bear]); 
    d=r<c[0]; B[d]=34; O[d]=44
    l=(r>=c[0])&(r<c[1]); m=rs.normal(center,1.5,n); sp=rs.integers(10,17,n); B[l]=np.floor(m-sp/2)[l]; O[l]=B[l]+sp[l]
    h=(r>=c[1])&(r<c[2]); m2=rs.normal(center-1.5,3,n); sp2=rs.integers(14,31,n); B[h]=np.floor(m2-sp2/2)[h]; O[h]=B[h]+sp2[h]
    bu=(r>=c[2])&(r<c[3]); bb=rs.choice([42,43,44,45,46],n,p=[.3,.3,.2,.12,.08]); B[bu]=bb[bu]; O[bu]=B[bu]+rs.integers(10,21,n)[bu]
    be=r>=c[3]; bo=rs.choice([38,39,40,41,42],n,p=[.15,.2,.25,.2,.2]); O[be]=bo[be]; B[be]=O[be]-rs.integers(10,21,n)[be]
    return B,O
def pnl_vec(myB,myO,B,O,a,conv):
    Os=np.sort(O); Oc=np.cumsum(Os); Bs=np.sort(B)[::-1]; Bc=np.cumsum(Bs)
    nb=np.searchsorted(Os,myB,'right'); ns=np.searchsorted(-Bs,-myO,'right')
    sO=np.where(nb>0,Oc[np.clip(nb-1,0,None)],0.); sB=np.where(ns>0,Bc[np.clip(ns-1,0,None)],0.)
    if conv=='mine':   return nb*(a-myB)+ns*(myO-a)
    if conv=='theirs': return (nb*a-sO)+(sB-ns*a)
    return (nb*a-(nb*myB+sO)/2)+((ns*myO+sB)/2-ns*a)
def pct(gB,gO,B,O,a,conv):
    mp=pnl_vec(gB,gO,B,O,a,conv); op=np.sort(pnl_vec(B,O,B,O,a,conv))
    return (1-(np.searchsorted(op,mp,'left')+np.searchsorted(op,mp,'right'))/2/len(op))*100*0+ (np.searchsorted(op,mp,'left')+np.searchsorted(op,mp,'right'))/2/len(op)*100
grid=np.array([(b,b+s) for b in range(28,47) for s in range(10,31)],float); gB,gO=grid[:,0],grid[:,1]
scen=[(bu,be) for bu in [0.02,0.05,0.08,0.12] for be in [0.0,0.02,0.04,0.06]]
res={}
S_=500
for bu,be in scen:
    tot={c:np.zeros(len(grid)) for c in ['mine','theirs','mid']}
    for _ in range(S_):
        a=A[rng.integers(len(A))]; c3=rng.uniform(41.5,44.5); B,O=field(517,rng,bu,be,c3)
        for c in tot: tot[c]+=pct(gB,gO,B,O,a,c)
    for c in tot: tot[c]/=S_
    res[(bu,be)]=tot
pickle.dump((grid,res),open("q2v2.pkl","wb"))
# summary tables
cands=[(34,44),(35,45),(36,46),(38,48),(39,49),(40,50),(41,51),(37,47),(33,43),(34,45),(35,46)]
print("E[percentile] by scenario (bull%%, bear%%), avg of 3 conventions.  columns = candidate quotes")
print("bull bear | "+" ".join("%6s"%("%d/%d"%c) for c in cands))
for (bu,be),tot in res.items():
    avg=sum(tot.values())/3
    print("%4.0f %4.0f | "%(bu*100,be*100)+" ".join("%6.1f"%avg[np.where((gB==b)&(gO==o))[0][0]] for b,o in cands))
# robust: average over all scenarios and conventions; and maximin
allavg=np.mean([sum(t.values())/3 for t in res.values()],0); allmin=np.min([np.min(np.stack(list(t.values())),0) for t in res.values()],0)
k=np.argsort(-allavg)[:8]; print("\nROBUST (avg over 16 scenarios x 3 conv):",[((int(gB[i]),int(gO[i])),round(allavg[i],1)) for i in k])
k=np.argsort(-allmin)[:8]; print("MAXIMIN:",[((int(gB[i]),int(gO[i])),round(allmin[i],1)) for i in k])
print("per-convention robust argmax:")
for c in ['mine','theirs','mid']:
    v=np.mean([t[c] for t in res.values()],0); k=np.argsort(-v)[:5]; print("   %-6s %s"%(c,[((int(gB[i]),int(gO[i])),round(v[i],1)) for i in k]))
