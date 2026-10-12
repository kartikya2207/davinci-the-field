import numpy as np
from field_core import *
from mc import s_with_me, MYW
np.set_printoptions(precision=3, suppress=True)
D=np.load("mc_out.npz"); names="RPS"; MY=TABLES[MY_TABLE]
EDGES=[30.,34.,38.,41.]
BNAME=["<30","30-34","34-38","38-41","41+"]

def buckets(sS_pct, upper_inclusive=False):
    if upper_inclusive:   # (.,30], (30,34], (34,38], (38,41], (41,.)
        return np.searchsorted(EDGES, sS_pct, side='left')
    return np.searchsorted(EDGES, sS_pct, side='right')   # [.,30),[30,34),... default

print("="*74); print("PART 2. JOINT DISTRIBUTION OF (sR,sP,sS)   [percent]")
for k in ["A_user_levelk","B_mine_levelk","C_ch","BLEND"]:
    s=D[k]*100
    print(f"\n-- {k}  N={len(s)}")
    for i,nm in enumerate("sR sP sS".split()):
        q=np.percentile(s[:,i],[5,25,50,75,95])
        print(f"   {nm}: mean {s[:,i].mean():6.2f} sd {s[:,i].std():5.2f} | p5 {q[0]:5.1f} p25 {q[1]:5.1f} p50 {q[2]:5.1f} p75 {q[3]:5.1f} p95 {q[4]:5.1f}")
    print(f"   corr(sP,sS)={np.corrcoef(s[:,1],s[:,2])[0,1]:+.3f}  corr(sR,sS)={np.corrcoef(s[:,0],s[:,2])[0,1]:+.3f}  corr(sR,sP)={np.corrcoef(s[:,0],s[:,1])[0,1]:+.3f}")

print("\n"+"="*74); print("GAME 3 BUCKET PROBABILITIES  (sS settles INCLUDING my own 100 throws, me all-in Scissors)")
for k in ["A_user_levelk","B_mine_levelk","C_ch","BLEND"]:
    sS=s_with_me(D[k],[0,0,1])[:,2]*100
    for ui,lab in [(False,"[lo,hi) lower-inclusive"),(True,"(lo,hi] upper-inclusive")]:
        b=buckets(sS,ui); p=np.bincount(b,minlength=5)/len(b)
        if not ui: print(f"\n-- {k}")
        print(f"   {lab}: "+"  ".join(f"{BNAME[i]} {100*p[i]:5.2f}%" for i in range(5))+f"   modal={BNAME[p.argmax()]}")
    p=np.bincount(buckets(sS),minlength=5)/len(sS)
    o=np.argsort(p)[::-1]
    print(f"   modal margin: {BNAME[o[0]]} beats {BNAME[o[1]]} by {100*(p[o[0]]-p[o[1]]):.2f} pts")
    print(f"   P(sS within 0.5pp of the 34 boundary) = {100*np.mean(np.abs(sS-34)<0.5):.2f}%")
    print(f"   expected Game-3 score (200-2*pts_in_true_bucket) for all-in on each bucket:")
    print("      "+"  ".join(f"{BNAME[i]} {200-200*p[i]:6.2f}" for i in range(5)))

print("\n"+"="*74); print("GAME 3: is all-in really optimal? (score = 200 - 2*points_in_true_bucket)")
p=np.bincount(buckets(s_with_me(D['BLEND'],[0,0,1])[:,2]*100),minlength=5)/len(D['BLEND'])
print("  E[score(a)] = 200 - 2*sum_i a_i p_i  -> LINEAR in the allocation a, so the minimiser")
print("  puts all 100 on argmax_i p_i. Verified numerically against 20000 random allocations:")
rng=np.random.default_rng(1); A=rng.dirichlet(np.ones(5)*0.7,20000)*100
sc=200-2*(A@p); best_allin=200-200*p.max()
print(f"   best random allocation {sc.min():.4f} vs all-in-on-mode {best_allin:.4f}  -> all-in wins: {sc.min()>=best_allin-1e-9}")

print("\n"+"="*74); print("GAME 1: expected score of each corner (MY table 3/5/9, my 0.5% self-effect INCLUDED)")
for k in ["A_user_levelk","B_mine_levelk","C_ch","BLEND"]:
    print(f"\n-- {k}")
    row=[]
    for m in range(3):
        w=np.zeros(3); w[m]=1
        s=s_with_me(D[k],w); v=MY*multipliers(s)
        sc=100*v[:,m]/v.max(axis=1)
        row.append(sc)
        print(f"   all-in {names[m]}: E[score]={sc.mean():7.3f}  P(score==100)={100*np.mean(sc>99.999999):6.2f}%  p5={np.percentile(sc,5):6.2f} median={np.median(sc):6.2f}")
    # P(move m is THE argmax) evaluated at the shares that move produces
    v0=MY*multipliers(s_with_me(D[k],[1/3,1/3,1/3]))
    am=v0.argmax(axis=1); pa=np.bincount(am,minlength=3)/len(am)
    print(f"   P(argmax is R/P/S), evaluated at neutral shares: {np.round(100*pa,2)}")
