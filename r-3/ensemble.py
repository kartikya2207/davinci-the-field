import numpy as np
from core import *
rng=np.random.default_rng(99)
m4=np.load("mc4_out_s0.npy"); m3=np.load("mc3_out_s0.npy"); tr=np.load("trend_blend_3col.npy")
# weights justified by day-2 backtest: mc-family retrodicted 42.3 (miss -1.0); trend family cannot be backtested (needs 2 points).
# mc4 carries the prompt-variant mechanism absent from mc3 and trend.  weights: mc4 .5, mc3 .25, trend .25
n=200000; k=rng.choice(3,n,p=[.5,.25,.25])
rows=[m4[rng.integers(len(m4),size=n)],m3[rng.integers(len(m3),size=n)],tr[rng.integers(len(tr),size=n)]]
E=np.where(k[:,None]==0,rows[0],np.where(k[:,None]==1,rows[1],rows[2]))
# trend rows have no R/P: fill from mc4 rows conditioned loosely (only S is used downstream for Q2/Q3/Q4)
E[k==2,0]=rows[0][k==2,0]; E[k==2,1]=rows[0][k==2,1]
np.save("ensemble_out.npy",E)
S=E[:,2]*100
print("ENSEMBLE predictive S3: mean %.1f median %.1f sd %.2f  5/25/75/95 = %s"%(S.mean(),np.median(S),S.std(),np.round(np.percentile(S,[5,25,75,95]),1)))
print("  buckets:",np.round(np.bincount(bucket(S),minlength=5)/n*100,1)," within±1 of 41: %.1f%%  within±1 of 38: %.1f%%"%(100*((S>=40)&(S<42)).mean(),100*((S>=37)&(S<39)).mean()))
print("  P(S>=41)=%.1f  P(S>40)=%.1f  P(S>=40)=%.1f"%(100*(S>=41).mean(),100*(S>40).mean(),100*(S>=40).mean()))
