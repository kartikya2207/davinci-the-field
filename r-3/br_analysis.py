import numpy as np
from core import *
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs])
def br_table(anchor):
    out={}; agg=np.zeros(3)
    for t,w in zip(tabs,wts):
        v=values(t,anchor); m=int(np.argmax(v)); out[t]=(m,v); agg[m]+=w
    return out,agg
for name,anc in [("uniform",UNI),("Day1",D1),("Day2",D2),("L1 fixed pt",L1)]:
    br,agg=br_table(anc)
    print("\nBR to %s %s  -> aggregate if all play it: R%.0f P%.0f S%.0f"%(name,np.round(anc*100,1),*agg*100))
    for t in tabs:
        m,v=br[t]; s=sorted(v)[::-1]
        print("   %s -> %s  values %s  margin over 2nd = %.1f%%"%(t,"RPS"[m],np.round(v,3),100*(s[0]/s[1]-1)))
