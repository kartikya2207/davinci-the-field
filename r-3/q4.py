import numpy as np, itertools, sys
from core import *
f=sys.argv[1] if len(sys.argv)>1 else "mc3_out_s0.npy"
A=np.load(f)[:,2]*100
res=[]
for dec in itertools.product([1,-1,0],repeat=4):
    sc=q4_score(dec,A); res.append((sc.mean(),np.percentile(sc,5),q4_profit(dec,A).mean(),dec))
res.sort(reverse=True)
lab={1:"Buy",-1:"Sell",0:"Leave"}
print("Q4 on %s (mean S=%.1f): top 8 by E[score]"%(f,A.mean()))
for m,p5,ep,d in res[:8]: print("   %-28s E=%.2f p5=%.1f E[profit]=%.1f"%("/".join(lab[x] for x in d),m,p5,ep))
for d in [(1,1,1,0),(1,1,1,1)]:
    for sh in [-3,0,3]:
        print("   %s at forecast%+d: E=%.2f"%("/".join(lab[x] for x in d),sh,q4_score(d,A+sh).mean()))
print("   P(A>40)=%.3f  P(A>41)=%.3f  P(A<40)=%.3f"%((A>40).mean(),(A>41).mean(),(A<40).mean()))
