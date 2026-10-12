import numpy as np, itertools
from core import TABLES, values
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs])
D1=np.array([30.7,32.5,36.9])/100

def lvl_aggs(anchor,kmax=4):
    aggs=[np.array([1/3,1/3,1/3])]; bel=np.array(anchor,float)
    for k in range(1,kmax+1):
        a=np.zeros(3)
        for t,w in zip(tabs,wts):
            a[int(np.argmax(values(t,bel)))]+=w
        aggs.append(a); bel=a
    return np.array(aggs)

# --- DAY 1: no feedback, anchor = uniform thirds
A1=lvl_aggs([1/3,1/3,1/3],3)
best=None
for f0 in np.arange(0,1.001,0.005):
  for f1 in np.arange(0,1.001-f0+1e-9,0.005):
    for f2 in np.arange(0,1.001-f0-f1+1e-9,0.005):
      f3=1-f0-f1-f2
      if f3<-1e-9: continue
      m=f0*A1[0]+f1*A1[1]+f2*A1[2]+f3*A1[3]
      e=np.abs(m-D1).sum()
      if best is None or e<best[0]: best=(e,f0,f1,f2,f3,m)
e,f0,f1,f2,f3,m=best
print("Day1 level aggregates:\n",np.round(A1*100,1))
print("BEST FIT L0=%.3f L1=%.3f L2=%.3f L3=%.3f  -> %s  (target %s) L1err=%.4f"%(f0,f1,f2,f3,np.round(m*100,2),D1*100,e))

# two-type restricted fit (L0/L1 only) for comparison
f=np.arange(0,1.001,0.001); mm=np.outer(1-f,A1[0])+np.outer(f,A1[1])
err=np.abs(mm-D1).sum(1); i=err.argmin()
print("L0/L1 only: f1=%.3f -> %s err=%.4f"%(f[i],np.round(mm[i]*100,2),err[i]))
np.save("A1.npy",A1)
