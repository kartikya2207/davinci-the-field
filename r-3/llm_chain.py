"""What does a mid-effort LLM tell each table TODAY, given it can see 41.3?
Three prompt variants a student might paste:  (V1) question only, no results -> LLM assumes equal thirds / worked example
(V2) question + Day-2 aggregate -> LLM computes values at D2   (V3) question + both days + trend -> LLM extrapolates to ~R21/P33/S45 then computes.
For each variant and table, the arithmetic is deterministic; the ONLY judgment is what an LLM does with a near-tie."""
import numpy as np
from core import *
tabs=list(TABLES); wts=np.array([TABLES[t] for t in tabs])
EXTRAP=np.array([0.21,0.33,0.45])   # a plausible LLM linear-trend anticipation
def chain(t,anchor):
    v=values(t,anchor); o=np.argsort(-v); m=v[o[0]]/v[o[1]]-1
    if m<0.05:   return "RPS"[o[0]], "near-tie (%.1f%%): LLM likely says '%s slight edge' or a mix"%(100*m,"RPS"[o[0]])
    return "RPS"[o[0]], "clear (%.0f%% margin)"%(100*m)
print("%-9s | %-30s | %-30s | %-30s"%("table","V1 question only (uniform)","V2 + Day-2 aggregate","V3 + trend extrapolation"))
agg={k:np.zeros(3) for k in ["V1","V2","V3"]}
for t,w in zip(tabs,wts):
    c1=chain(t,UNI); c2=chain(t,D2); c3=chain(t,EXTRAP)
    print("%-9s | %s %-28s | %s %-28s | %s %-28s"%(t,c1[0],c1[1],c2[0],c2[1],c3[0],c3[1]))
    for k,c in zip(["V1","V2","V3"],[c1,c2,c3]): agg[k]["RPS".index(c[0])]+=w
for k in agg: print("bloc aggregate if ALL LLM users pasted %s: R%.0f P%.0f S%.0f"%(k,*agg[k]*100))
# The (5,9,3) near-tie under V2 is the whole Q1 question. Under V3 (trend), Rock wins clearly:
print("\n(5,9,3) at V3 anchor:",np.round(values((5,9,3),EXTRAP),3)," -> Rock margin %.0f%%"%(100*(values((5,9,3),EXTRAP)[0]/values((5,9,3),EXTRAP)[1]-1)))
print("(5,3,9) at V3 anchor:",np.round(values((5,3,9),EXTRAP),3)," -> S vs R margin %.0f%%"%(100*(values((5,3,9),EXTRAP)[2]/values((5,3,9),EXTRAP)[0]-1)))
print("(3,5,9) at V3 anchor:",np.round(values((3,5,9),EXTRAP),3))
# My own table under the V2/V3 BLOC aggregate + L0 dilution 20%:
for k in ["V2","V3"]:
    s=0.8*agg[k]+0.2*UNI; v=values(MINE,s); print("if bloc=%s with 20%% L0: campus R%.0f P%.0f S%.0f -> my values %s best=%s"%(k,*s*100,np.round(v,2),"RPS"[v.argmax()]))
