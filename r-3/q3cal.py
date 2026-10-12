"""Calibrate the Q3 opponent-field model against Day-2 ground truth:
my (0,0,55,40,5) at actual 41.3 (bucket >=41) scored 37.1.  Day-1: all-in 34-38 at 36.9 scored ~95-100 (by subtraction).
Day-2 field: defaults / LLM bloc (day-2 advice centred 34-38 & 38-41) / sharp / boundary-splitters (38 boundary) / stale."""
import numpy as np
from core import *
rng=np.random.default_rng(0)
def day2_field(n,rs,fr):
    E=np.zeros((n,5)); cnt=rs.multinomial(n,fr); i=0
    E[i:i+cnt[0]]=20; i+=cnt[0]                                            # defaults
    k=cnt[1]; top=rs.choice([50,60,70,80,100]); b4=rs.choice([0,10,20,30,40]); b4=min(b4,100-top); b2=rs.choice([0,10,20]); b2=min(b2,100-top-b4)
    E[i:i+k]=[0,b2,top,b4,100-top-b4-b2]; i+=k                              # LLM bloc: modal 34-38 (research-pass style)
    k=cnt[2]; E[i:i+k]=rs.dirichlet(np.array([0.3,2,8,8,3])*rs.uniform(.5,3),k)*100; i+=k   # sharp: 34-41 spread
    k=cnt[3]; sp=rs.choice([40,50,60],k); E[i:i+k,2]=100-sp; E[i:i+k,3]=sp; i+=k         # 38-boundary splitters
    k=cnt[4]; E[i:i+k]=rs.dirichlet(np.array([2,5,6,3,1])*rs.uniform(.5,2),k)*100; i+=k  # stale/low
    return np.round(E)
def pct(alloc,E,tb):
    od=2*(100-E[:,tb]); md=2*(100-alloc[tb]); return ((od>md).mean()+0.5*(od==md).mean())*100
print("Day-2 calibration of my (0,0,55,40,5) at bucket >=41 (obs 37.1) and Day-1 all-in 34-38 at bucket 34-38 (obs ~95-100):")
for fr in [[.30,.35,.15,.10,.10],[.20,.45,.15,.10,.10],[.25,.30,.25,.10,.10],[.15,.35,.30,.10,.10],[.35,.25,.20,.10,.10]]:
    p2=np.mean([pct(np.array([0,0,55,40,5]),day2_field(517,rng,fr),4) for _ in range(300)])
    p1=np.mean([pct(np.array([0,0,100,0,0]),day2_field(466,rng,fr),2) for _ in range(300)])
    pd=np.mean([pct(np.array([20,20,20,20,20]),day2_field(517,rng,fr),4) for _ in range(300)])
    print("  fractions def/LLM/sharp/split/stale=%s -> day2 %.1f (37.1) | day1 %.1f (~95+) | default on day2 would score %.1f"%(fr,p2,p1,pd))
