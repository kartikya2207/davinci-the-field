"""Fit level mixtures to BOTH days. Report identifiability honestly."""
import numpy as np
from core import *
from math import exp, factorial
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs])
def agg_br(anchor):
    a=np.zeros(3)
    for t,w in zip(tabs,wts): a[int(np.argmax(values(t,anchor)))]+=w
    return a
L1a=agg_br(UNI); L2a=agg_br(L1a); L3a=agg_br(L2a)
print("ladder: L1",L1a*100,"L2",L2a*100,"L3",L3a*100)
assert np.allclose(agg_br(D1),L1a)          # BR(D1)==BR(unif): feedback could not rotate on day 2
print("BR(D2) aggregate =",agg_br(D2)*100,"  <-- DIFFERENT from L1: day-3 feedback DOES rotate")

# ---- Model M1: one parameter per day, f = share at L1, rest uniform ----
def fit_f(D):
    f=np.linspace(0,1,10001); M=np.outer(1-f,UNI)+np.outer(f,L1a)
    e=((M-D)**2).sum(1); i=e.argmin(); return f[i],np.sqrt(e[i]),M[i]
for nm,D in [("Day1",D1),("Day2",D2)]:
    f,e,m=fit_f(D); print("M1 %s: f=%.3f rmse=%.4f fitted=%s obs=%s"%(nm,f,e,np.round(m*100,1),D*100))

# ---- Model M2: two params per day (L1 share f, L2 share g), rest uniform: EXACT per day ----
def fit_fg(D):
    best=None
    for f in np.linspace(0,1,1001):
        for g in np.linspace(0,1-f,int((1-f)*1000)+1):
            m=(1-f-g)*UNI+f*L1a+g*L2a; e=np.abs(m-D).sum()
            if best is None or e<best[0]: best=(e,f,g,m)
    return best
for nm,D in [("Day1",D1),("Day2",D2)]:
    e,f,g,m=fit_fg(D); print("M2 %s: L0=%.3f L1=%.3f L2=%.3f  L1err=%.4f fitted=%s"%(nm,1-f-g,f,g,e,np.round(m*100,2)))

# ---- Model M3: Poisson-CH tau per day (1 param) ----
def ch(tau,K=4):
    w=np.array([exp(-tau)*tau**k/factorial(k) for k in range(K)]); return w/w.sum()
lad=np.array([UNI,L1a,L2a,L3a])
for nm,D in [("Day1",D1),("Day2",D2)]:
    ts=np.linspace(0.01,5,5000); e=[np.abs(ch(t)@lad-D).sum() for t in ts]; i=int(np.argmin(e))
    print("M3 %s: tau=%.3f L1err=%.4f fitted=%s"%(nm,ts[i],e[i],np.round(ch(ts[i])@lad*100,2)))

# ---- Model M4: L0 + naive-L1 + INERTIA(previous aggregate) + L2  for day 2 ----
# day-2 = a*UNI + b*L1 + c*D1 + d*L2 ; 3 params free, 2 obs -> family of solutions. Show the ridge.
print("\nM4 day2 = a*UNI + b*L1 + c*D1(inertia) + d*L2, sweep c:")
for c in [0,0.1,0.2,0.3,0.4]:
    best=None
    for b in np.linspace(0,1-c,501):
        for d in np.linspace(0,1-c-b,int((1-c-b)*500)+1):
            a=1-b-c-d; m=a*UNI+b*L1a+c*D1+d*L2a; e=np.abs(m-D2).sum()
            if best is None or e<best[0]: best=(e,a,b,d)
    e,a,b,d=best; print("   inertia c=%.1f -> L0=%.3f L1=%.3f L2=%.3f err=%.4f"%(c,a,b,d,e))
