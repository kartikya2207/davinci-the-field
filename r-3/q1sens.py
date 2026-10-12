"""Q1: P(Paper best) and E[v/vmax] as a function of the recompute fraction r and LLM bloc size."""
import numpy as np
from core import *
par=np.load("mc3_par_s0.npy"); V=np.load("mc3_V_s0.npy"); out=np.load("mc3_out_s0.npy")
rat=V/V.max(1,keepdims=True); best=V.argmax(1)
llm,rho,inert,r_llm,l2,mom=par.T
print("Q1 sensitivity (seed 0, 100k). Columns: P(Paper best) | E[v/vmax] S | E[v/vmax] P | mean sP | mean sR")
print("by r_llm (P an LLM answer recomputes vs published D2):")
for lo,hi in [(0,.2),(.2,.35),(.35,.5),(.5,.65),(.65,.8),(.8,1)]:
    m=(r_llm>=lo)&(r_llm<hi); print("   r_llm %.2f-%.2f  n=%5d : %5.1f%% | %5.1f | %5.1f | sP %.1f sR %.1f"%(lo,hi,m.sum(),100*(best[m]==1).mean(),100*rat[m,2].mean(),100*rat[m,1].mean(),100*out[m,1].mean(),100*out[m,0].mean()))
print("by LLM bloc size:")
for lo,hi in [(.2,.35),(.35,.5),(.5,.65),(.65,.8)]:
    m=(llm>=lo)&(llm<hi); print("   llm %.2f-%.2f : %5.1f%% | %5.1f | %5.1f"%(lo,hi,100*(best[m]==1).mean(),100*rat[m,2].mean(),100*rat[m,1].mean()))
print("joint: bloc>=50%% AND r_llm>=0.5 : P(Paper best)=%.1f%%  E S=%.1f  E P=%.1f"%tuple([100*(best[(llm>=.5)&(r_llm>=.5)]==1).mean(),100*rat[(llm>=.5)&(r_llm>=.5),2].mean(),100*rat[(llm>=.5)&(r_llm>=.5),1].mean()]))
print("joint: bloc<50%% OR  r_llm<0.5  : P(Paper best)=%.1f%%  E S=%.1f  E P=%.1f"%tuple([100*(best[~((llm>=.5)&(r_llm>=.5))]==1).mean(),100*rat[~((llm>=.5)&(r_llm>=.5)),2].mean(),100*rat[~((llm>=.5)&(r_llm>=.5)),1].mean()]))
# break-even: what P(Paper best) makes pure P == pure S ?
print("\nconditional payoffs: E[S ratio | P best]=%.1f  E[P ratio | S best]=%.1f"%(100*rat[best==1,2].mean(),100*rat[best==2,1].mean()))
a=rat[best==1,2].mean(); b=rat[best==2,1].mean()
pb=(1-b)/((1-a)+(1-b)); print("break-even P(Paper best) for pure-P to beat pure-S = %.1f%%"%(100*pb))
# mixes
for w in [(0,0,100),(0,20,80),(0,30,70),(0,50,50),(0,70,30),(0,100,0)]:
    ww=np.array(w)/100; sc=100*(rat@ww); print("   mix R/P/S=%s : E=%.2f  p10=%.1f  p90=%.1f  P(score<70)=%.1f%%"%(w,sc.mean(),np.percentile(sc,10),np.percentile(sc,90),100*(sc<70).mean()))
