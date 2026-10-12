import numpy as np, itertools
from core import values, score_q1
out=np.load("mc_out.npy"); MINE=(3,5,9); T=len(out)
V=np.array([values(MINE,s) for s in out])            # (T,3)
am=V.argmax(1)
print("Q1  P(argmax=R/P/S) = %s"%np.round(np.bincount(am,minlength=3)/T*100,3))
rat=V/V.max(1,keepdims=True)
print("Q1  E[v_m/vmax] R/P/S = %s  -> pure best = %s"%(np.round(rat.mean(0)*100,3),"RPS"[rat.mean(0).argmax()]))
for w,lab in [((0,0,1),"0/0/100 S"),((0,10,90),"0/10/90"),((10,0,90),"10/0/90"),((0,20,80),"0/20/80"),((5,5,90),"5/5/90"),((0,0,100),"pure S")]:
    ww=np.array(w,float); ww/=ww.sum()
    sc=100*(rat*ww).sum(1); print("   %-10s E=%.3f  p5=%.2f  min=%.2f"%(lab,sc.mean(),np.percentile(sc,5),sc.min()))
# exact flip surface for MY table
print("\nQ1 FLIP THRESHOLDS (my table 3/5/9):")
for sP in [0.24,0.28,0.30,0.32,0.34]:
    lo=None
    for sS in np.arange(0.30,0.80,0.0005):
        sR=1-sP-sS
        if sR<0: break
        v=values(MINE,(sR,sP,sS))
        if v[0]>v[2] and lo is None: lo=sS
    print("   sP=%.2f : S->R flip at sS=%s"%(sP,("%.3f"%lo) if lo else "never"))
for sP in [0.20,0.25,0.30]:
    lo=None
    for sR in np.arange(0.20,0.80,0.0005):
        sS=1-sP-sR
        if sS<0: break
        v=values(MINE,(sR,sP,sS))
        if v[1]>v[2] and lo is None: lo=sR
    print("   sP=%.2f : S->P flip at sR=%s"%(sP,("%.3f"%lo) if lo else "never"))

# ---------------- Q4 : all 81 combos scored by SCORE not profit ----------------
A=out[:,2]*100
MK=[(5,10),(15,20),(25,30),(35,40)]
def prof(dec,a):
    p=np.zeros_like(a)
    for d,(b,o) in zip(dec,MK):
        if d==1: p=p+(a-o)
        elif d==-1: p=p+(b-a)
    return p
avail=np.zeros_like(A)
for b,o in MK: avail=avail+np.maximum(np.maximum(A-o,b-A),0)
res=[]
for dec in itertools.product([1,-1,0],repeat=4):
    p=prof(dec,A); sc=np.clip(100*p/np.where(avail>0,avail,1e-9),0,100)
    res.append((sc.mean(),np.percentile(sc,5),p.mean(),dec))
res.sort(reverse=True)
print("\nQ4 TOP 8 by EXPECTED SCORE   (dec: 1=Buy -1=Sell 0=Leave, order M1..M4)")
for m,p5,ep,d in res[:8]: print("   %s  E[score]=%.3f  p5=%.1f  E[profit]=%.2f"%(str(d),m,p5,ep))
print("   research pick (1,1,1,-1): E[score]=%.3f"%[r[0] for r in res if r[3]==(1,1,1,-1)][0])
print("   P(actual>40)=%.4f  P(actual<35)=%.4f"%((A>40).mean(),(A<35).mean()))

# ---------------- Q3 : rank-based simulation ----------------
P=np.load("bucketP.npy"); print("\nQ3 bucket P = %s"%np.round(P*100,2))
print("Q3 raw-deviation-optimal = all-in on bucket %d (%s)"%(P.argmax()+1,["<30","30-34","34-38","38-41",">=41"][P.argmax()]))
