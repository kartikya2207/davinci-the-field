import numpy as np, pickle, itertools
from core import values
out=np.load("mc_out.npy"); A0=out[:,2]*100
# ---- STRESS: shift the whole predictive distribution to the research pass's mean 37
for shift,lab in [(0,"my MC mean 38.4"),(-1.4,"research mean 37.0"),(-2.4,"pessimistic 36.0"),(+2.0,"scissors cascade 40.4")]:
    A=A0+shift
    b=[(A<30),(A>=30)&(A<34),(A>=34)&(A<38),(A>=38)&(A<41),(A>=41)]
    P=np.array([x.mean() for x in b])
    # Q4
    MK=[(5,10),(15,20),(25,30),(35,40)]
    av=np.zeros_like(A)
    for lo,hi in MK: av=av+np.maximum(np.maximum(A-hi,lo-A),0)
    best=[]
    for dec in itertools.product([1,-1,0],repeat=4):
        p=np.zeros_like(A)
        for d,(lo,hi) in zip(dec,MK):
            if d==1:p=p+(A-hi)
            elif d==-1:p=p+(lo-A)
        best.append((np.clip(100*p/np.maximum(av,1e-9),0,100).mean(),dec))
    best.sort(reverse=True)
    print("%-26s mean=%.1f buckets=%s | Q4 best=%s(%.1f) BBBL=%.1f BBBS=%.1f"%(
        lab,A.mean(),np.round(P*100,1),best[0][1],best[0][0],
        [s for s,d in best if d==(1,1,1,0)][0],[s for s,d in best if d==(1,1,1,-1)][0]))
