import numpy as np, itertools
from core import TABLES, values
rng=np.random.default_rng(7)
N=467
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs])

def agg_of_level(level, prev):
    """aggregate shares if EVERY agent plays at `level` given belief `prev`."""
    a=np.zeros(3)
    for t,w in zip(tabs,wts):
        b=np.array(prev)
        for _ in range(level):   # iterate best response `level` times
            v=values(t,b); m=int(np.argmax(v))
            b=np.zeros(3); b[m]=1.0   # placeholder, replaced below
        a+= w*_br_vec(t,prev,level)
    return a/a.sum()

def _br_vec(t,prev,level):
    """level-k BR for table t. level0 = prev-agnostic uniform."""
    if level==0: return np.array([1/3,1/3,1/3])
    b=np.array(prev,float)
    for k in range(level):
        # aggregate of everyone playing level k-1 against belief `prev`
        pass
    return None

# cleaner: build level aggregates iteratively
def level_aggregates(anchor, kmax=5):
    """anchor = belief L1 responds to (the Day-1 published aggregate).
       L0 = uniform thirds. Lk = BR to the aggregate produced if all played L(k-1)."""
    aggs=[np.array([1/3,1/3,1/3])]          # L0 aggregate
    beliefs=[np.array(anchor,float)]        # L1 responds to anchor
    choices=[None]
    for k in range(1,kmax+1):
        bel=beliefs[-1]; a=np.zeros(3); ch={}
        for t,w in zip(tabs,wts):
            v=values(t,bel); m=int(np.argmax(v)); ch[t]="RPS"[m]; a[m]+=w
        aggs.append(a); choices.append(ch); beliefs.append(a)
    return aggs, choices

if __name__=="__main__":
    d1=(0.307,0.325,0.369)
    aggs,ch=level_aggregates(d1,4)
    for k,a in enumerate(aggs):
        print("L%d aggregate R/P/S = %s"%(k,np.round(a*100,1)), ch[k] if k>0 else "(uniform)")
