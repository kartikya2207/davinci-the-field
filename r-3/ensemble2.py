"""JOB 2: re-weighted ensemble: trend/recruitment .6, MC v2 .25, MC v1 .15."""
import numpy as np
from core import *
rng=np.random.default_rng(7)
m4=np.load("mc4_out_s0.npy"); m3=np.load("mc3_out_s0.npy"); tr=np.load("trend_blend_3col.npy")
n=200000; k=rng.choice(3,n,p=[.25,.15,.60])
rows=[m4[rng.integers(len(m4),size=n)],m3[rng.integers(len(m3),size=n)],tr[rng.integers(len(tr),size=n)]]
E=np.where(k[:,None]==0,rows[0],np.where(k[:,None]==1,rows[1],rows[2])); E[k==2,0]=rows[0][k==2,0]; E[k==2,1]=rows[0][k==2,1]
np.save("ensemble2_out.npy",E); S=E[:,2]*100
print("RE-WEIGHTED ENSEMBLE (trend .6 / v2 .25 / v1 .15): mean %.1f median %.1f sd %.2f 5/95 %s"%(S.mean(),np.median(S),S.std(),np.round(np.percentile(S,[5,95]),1)))
print("  buckets:",np.round(np.bincount(bucket(S),minlength=5)/n*100,1),"  P(S>=45)=%.1f P(S>=44)=%.1f P(38-41)=%.1f"%(100*(S>=45).mean(),100*(S>=44).mean(),100*((S>=38)&(S<41)).mean()))
S0=np.load("ensemble_out.npy")[:,2]*100
print("ORIGINAL ENSEMBLE (v2 .5 / v1 .25 / trend .25): mean %.1f  P(38-41)=%.1f"%(S0.mean(),100*((S0>=38)&(S0<41)).mean()))
