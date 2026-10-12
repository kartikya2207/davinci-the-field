"""Q3 percentile optimiser. Opponent field = mixture of blocs; parameters randomised per trial.
usage: python q3rank.py <mc_out.npy> <seed> <R trials> [oppmodel]"""
import numpy as np, sys, itertools, pickle
from core import *
f=sys.argv[1]; seed=int(sys.argv[2]); R=int(sys.argv[3]); model=sys.argv[4] if len(sys.argv)>4 else "A"
rng=np.random.default_rng(seed)
S=np.load(f)[:,2]*100; TB=bucket(S)
def opp_field(n,rs,model):
    E=np.zeros((n,5)); i=0
    if model=="A":      # default / LLM-bloc(correlated) / sharp / anchored-split / inertial
        fr=rs.dirichlet([3,5,2,2,2])
    elif model=="B":    # bigger LLM bloc, fewer defaults
        fr=rs.dirichlet([2,8,2,2,1])
    else:               # C: field learned hard from day 2 -> heavy on >=41
        fr=rs.dirichlet([2,4,5,2,1])
    cnt=rs.multinomial(n,fr)
    # 1 defaults
    E[i:i+cnt[0]]=20; i+=cnt[0]
    # 2 LLM bloc: ONE correlated advice draw for the whole bloc, + small jitter subgroups
    top=rs.choice([50,60,65,70,75,80,90,100],p=[.08,.15,.12,.2,.15,.15,.1,.05])
    mid=rs.choice([0,10,20,25,30,35,40,50]); mid=min(mid,100-top)
    rest=100-top-mid
    base=np.array([0,0,rest,mid,top],float)
    k=cnt[1]; E[i:i+k]=base
    j=int(k*rs.uniform(0.2,0.5))                 # sub-bloc got a different draw
    top2=rs.choice([40,50,60,70,80]); mid2=min(rs.choice([20,30,40,50]),100-top2)
    E[i:i+j]=[0,0,100-top2-mid2,mid2,top2]; i+=k
    # 3 sharp: dirichlet centred on (0,0,3,22,75)
    k=cnt[2]; E[i:i+k]=rs.dirichlet(np.array([0.3,0.3,2,8,22])*rs.uniform(0.5,3),k)*100; i+=k
    # 4 anchored-split on 41.3 boundary: 50/50 or 40/60 between 38-41 and >=41
    k=cnt[3]; sp=rs.choice([40,50,60],k); E[i:i+k,3]=100-sp; E[i:i+k,4]=sp; i+=k
    # 5 inertial / stale: day-2 style allocations centred 34-38 / 38-41
    k=cnt[4]; E[i:i+k]=rs.dirichlet(np.array([1,3,8,6,2])*rs.uniform(0.5,2),k)*100; i+=k
    return np.round(E)
step=5
CAND=np.array([c for c in itertools.product(range(0,101,step),repeat=4) if sum(c)<=100])
CAND=np.column_stack([100-CAND.sum(1),CAND])          # (b1,b2,b3,b4,b5)
CAND=CAND[(CAND[:,0]<=20)]                              # <30 bucket: cap to keep search sane
res=np.zeros(len(CAND))
for r in range(R):
    tb=TB[rng.integers(len(TB))]
    E=opp_field(N3-1,rng,model); od=2*(100-E[:,tb]); md=2*(100-CAND[:,tb])
    ods=np.sort(od)
    lo=np.searchsorted(ods,md,'left'); hi=np.searchsorted(ods,md,'right')
    res+=(1-(lo+hi)/2/len(ods))*100   # pct = share of field with WORSE (higher) deviation, ties half
res/=R; o=np.argsort(-res)
print("Q3 rank sim  file=%s model=%s seed=%d R=%d  (bucket P=%s)"%(f,model,seed,R,np.round(np.bincount(TB,minlength=5)/len(TB)*100,1)))
for i in o[:12]: print("   %s  E[pct]=%.2f"%(CAND[i],res[i]))
def show(t):
    m=np.where((CAND==np.array(t)).all(1))[0]
    if len(m): print("   ref %-22s -> %.2f"%(str(t),res[m[0]]))
for t in [(0,0,0,0,100),(0,0,0,20,80),(0,0,0,30,70),(0,0,0,40,60),(0,0,5,25,70),(0,0,0,50,50),(0,0,55,40,5),(20,20,20,20,20),(0,0,0,100,0)]: show(t)
pickle.dump((CAND,res),open("q3res_%s_%s_s%d.pkl"%(f.split('.')[0],model,seed),"wb"))
