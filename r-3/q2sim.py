"""Q2 market sim. Opponent quote model with a calibration step:
  Day2: I quoted 36/46, actual 41.3, scored 90.4.  Day1: 30/40, actual 36.9, Q2 ~61-66 (inferred).
Then optimise my Day-3 quote under all three execution conventions on the Day-3 S distribution."""
import numpy as np, sys, pickle
from core import *
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
def quotes(n,rs,center,agg):
    """center = where sharp/LLM players think S will land; agg = aggressiveness scale (0..1)."""
    r=rs.random(n); B=np.empty(n); O=np.empty(n)
    d=r<0.30; B[d]=34; O[d]=44                                       # default keepers
    l=(r>=0.30)&(r<0.70)                                             # LLM/sharp: centred on forecast, spread 10-16
    m=rs.normal(center,1.5,n); sp=rs.integers(10,17,n)
    B[l]=np.floor(m-sp/2)[l]; O[l]=B[l]+sp[l]
    h=(r>=0.70)&(r<0.90)                                             # humans: wide, anchored on last aggregate
    m2=rs.normal(center-1,3,n); sp2=rs.integers(14,31,n)
    B[h]=np.floor(m2-sp2/2)[h]; O[h]=B[h]+sp2[h]
    g=r>=0.90                                                        # aggressive: one-sided at/near forecast
    side=rs.random(n)<0.5+0.3*agg
    ab=np.floor(center+rs.normal(0,2,n)); ao=np.floor(center-rs.normal(0,2,n))
    B[g&side]=ab[g&side]; O[g&side]=B[g&side]+rs.integers(10,21,n)[g&side]
    O[g&~side]=ao[g&~side]; B[g&~side]=O[g&~side]-rs.integers(10,21,n)[g&~side]
    return B,O
def pnl_vec(myB,myO,B,O,a,conv):
    Os=np.sort(O); Oc=np.cumsum(Os); Bs=np.sort(B)[::-1]; Bc=np.cumsum(Bs)
    nb=np.searchsorted(Os,myB,'right'); ns=np.searchsorted(-Bs,-myO,'right')
    sO=np.where(nb>0,Oc[np.clip(nb-1,0,None)],0.); sB=np.where(ns>0,Bc[np.clip(ns-1,0,None)],0.)
    if conv=='mine':   return nb*(a-myB)+ns*(myO-a)
    if conv=='theirs': return (nb*a-sO)+(sB-ns*a)
    return (nb*a-(nb*myB+sO)/2)+((ns*myO+sB)/2-ns*a)
def pct(myB,myO,B,O,a,conv):
    mp=pnl_vec(np.atleast_1d(myB),np.atleast_1d(myO),B,O,a,conv); op=np.sort(pnl_vec(B,O,B,O,a,conv))
    lo=np.searchsorted(op,mp,'left'); hi=np.searchsorted(op,mp,'right'); return (lo+hi)/2/len(op)*100
# ---------- CALIBRATION ----------
print("CALIBRATION: Day-2 (36/46 @41.3 -> obs 90.4) and Day-1 (30/40 @36.9 -> obs ~61-66)")
best=[]
for center2 in [37,38,39,40,41]:
  for agg in [0.0,0.5,1.0]:
    for conv in ['mine','theirs','mid']:
        p2=np.mean([pct(36,46,*quotes(517,rng,center2,agg),41.3,conv)[0] for _ in range(60)])
        p1=np.mean([pct(30,40,*quotes(466,rng,center2-4,agg),36.9,conv)[0] for _ in range(60)])
        best.append((abs(p2-90.4)+0.5*abs(p1-63.5),center2,agg,conv,p2,p1))
best.sort()
for b in best[:10]: print("   err %.1f  crowd-centre %d  agg %.1f  conv %-6s -> day2 %.1f (90.4)  day1 %.1f (~63)"%b)
print("   consistent conventions among top-10:",sorted(set(b[3] for b in best[:10])))
# ---------- DAY-3 OPTIMISATION ----------
A=np.load(sys.argv[2] if len(sys.argv)>2 else "mc3_out_s0.npy")[:,2]*100
cal=list(best[0]); cal[2]=float(sys.argv[3]) if len(sys.argv)>3 else cal[2]; print("\nDay-3 opponent centre: crowd sees 41.3 and the trend -> centre ~ U(41,44); agg=%.1f"%cal[2])
grid=np.array([(b,b+s) for b in range(30,48) for s in range(10,31)],float); gB,gO=grid[:,0],grid[:,1]
S_=2500; tot={c:np.zeros(len(grid)) for c in ['mine','theirs','mid']}
for _ in range(S_):
    a=A[rng.integers(len(A))]; c3=rng.uniform(41,44); B,O=quotes(517,rng,c3,cal[2])
    for c in tot: tot[c]+=pct(gB,gO,B,O,a,c)
for c in tot:
    tot[c]/=S_; k=np.argsort(-tot[c])[:5]
    print("Q2 %-6s top: %s"%(c,[((int(gB[i]),int(gO[i])),round(tot[c][i],1)) for i in k]))
comb=sum(tot.values())/3; mn=np.min(np.stack(list(tot.values())),0)
k=np.argsort(-comb)[:6]; print("ROBUST avg: %s"%[((int(gB[i]),int(gO[i])),round(comb[i],1)) for i in k])
k=np.argsort(-mn)[:6];  print("MAXIMIN  : %s"%[((int(gB[i]),int(gO[i])),round(mn[i],1)) for i in k])
print("\nbid sweep (offer=bid+10):")
for b in range(34,46):
    i=np.where((gB==b)&(gO==b+10))[0][0]; print("   %d/%d : mine %.1f theirs %.1f mid %.1f"%(b,b+10,tot['mine'][i],tot['theirs'][i],tot['mid'][i]))
print("offer sweep at bid 40:")
for o in [50,52,55,58,60,65,70]:
    i=np.where((gB==40)&(gO==o))[0]
    if len(i): print("   40/%d : mine %.1f theirs %.1f mid %.1f"%(o,tot['mine'][i[0]],tot['theirs'][i[0]],tot['mid'][i[0]]))
pickle.dump((grid,tot),open("q2res.pkl","wb"))
