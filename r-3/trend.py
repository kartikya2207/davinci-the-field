"""Six trend models on f = fraction of uniform->L1 gap closed. Two points, honest uncertainty."""
import numpy as np
from core import *
rng=np.random.default_rng(42)
# f per day from the 1-param least-squares fit (M1) and from Rock/Scissors separately
def f_of(D):
    f=np.linspace(0,1.2,12001); M=np.outer(1-f,UNI)+np.outer(f,L1); return f[((M-D)**2).sum(1).argmin()]
f1,f2=f_of(D1),f_of(D2)
fR=[(1/3-D1[0])/(1/3-0.20),(1/3-D2[0])/(1/3-0.20)]; fS=[(D1[2]-1/3)/(0.46-1/3),(D2[2]-1/3)/(0.46-1/3)]
print("f (LS all three shares): day1 %.3f day2 %.3f | Rock-only %s | Scissors-only %s"%(f1,f2,np.round(fR,3),np.round(fS,3)))
def S_of_f(f): return 100*((1-f)/3+0.46*f)
T=200000
models={}
# 1 linear
f3=f2+(f2-f1); models["linear"]=np.clip(f3+rng.normal(0,0.06,T),0,1.05)
# 2 saturating: each round closes fraction k of remaining gap; k=(f2-f1)/(1-f1)
k=(f2-f1)/(1-f1); f3=f2+k*(1-f2); models["saturating(k=%.2f)"%k]=np.clip(f3+rng.normal(0,0.06,T),0,1.05)
# 3 logistic in logit(f)
lg=lambda x:np.log(x/(1-x)); il=lambda z:1/(1+np.exp(-z))
z3=lg(f2)+(lg(f2)-lg(f1)); models["logistic"]=np.clip(il(z3+rng.normal(0,0.35,T)),0,1.05)
# 4 level-mixture (M2 recruitment): L0 35% -> 35%*(1-rho), rho~U(.3,.65); f3 = 1-L0-L2
rho=rng.uniform(0.30,0.65,T); models["recruitment"]=1-0.35*(1-rho)-rng.uniform(0,0.06,T)
# 5 overshoot: + momentum L0 mass on Scissors
models["overshoot"]=models["recruitment"]+rng.uniform(0,0.12,T)
# 6 reversal: some S-holders defect to Paper (level-2), reduces S by 0-6 points
rev=models["recruitment"].copy()
print("\nModel            mean S3   median   5%%   95%%   P(>=41)  P(38-41)  P(within±1 of 41)")
allS=[]
for nm,f in models.items():
    S=S_of_f(f)+rng.normal(0,0.6,T)   # N=518 sampling noise (Bernoulli SD .5 /sqrt(518) ~2.2%->but shares of a share; ~0.6pt)
    if nm=="overshoot": S=np.minimum(S,49)
    if nm.startswith("recruitment"): S_rev=S-rng.uniform(0,6,T)*(rng.random(T)<0.5)
    p=lambda x:(x).mean()*100
    print("%-16s %6.1f  %6.1f  %5.1f %5.1f   %5.1f   %5.1f     %5.1f"%(nm,S.mean(),np.median(S),np.percentile(S,5),np.percentile(S,95),p(S>=41),p((S>=38)&(S<41)),p((S>=40)&(S<42))))
    allS.append(S)
S=S_rev; print("%-16s %6.1f  %6.1f  %5.1f %5.1f   %5.1f   %5.1f     %5.1f"%("reversal(L2 def)",S.mean(),np.median(S),np.percentile(S,5),np.percentile(S,95),p(S>=41),p((S>=38)&(S<41)),p((S>=40)&(S<42)))); allS.append(S)
# blend: equal weights on linear/saturating/logistic/recruitment + half on overshoot & reversal
W=[1,1,1,1,0.5,0.5]; idx=rng.choice(len(allS),T,p=np.array(W)/sum(W))
B=np.array([allS[j][i] for i,j in enumerate(idx)])
print("%-16s %6.1f  %6.1f  %5.1f %5.1f   %5.1f   %5.1f     %5.1f"%("BLEND",B.mean(),np.median(B),np.percentile(B,5),np.percentile(B,95),p(B>=41),p((B>=38)&(B<41)),p((B>=40)&(B<42))))
np.save("trend_blend_S.npy",B)
# Backtest of the trend models: fit on day 1 only -> predict day 2?  With ONE point no trend exists.
# The only day-1-only prediction is "recruitment at the same rate as ... unknown". Report the honest answer.
print("\nBACKTEST(trend): with only Day 1, f1=%.3f; predicting Day 2 requires a recruitment rate. rho=0.49 (unknown a priori) gives S2=%.1f; rho~U(.3,.65) prior gives %.1f-%.1f"%(
    f1, S_of_f(f1+0.49*(1-f1)), S_of_f(f1+0.30*(1-f1)), S_of_f(f1+0.65*(1-f1))))
