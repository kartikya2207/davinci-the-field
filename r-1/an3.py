import numpy as np, itertools
from field_core import *
from mc import s_with_me, MYW
np.set_printoptions(precision=3, suppress=True)
names="RPS"; MY=TABLES[MY_TABLE]; LK,_=level_profiles(6)
D=np.load("mc_out.npz")

print("="*78); print("GAME 1 FLIP: clean 2-D sweep, L3 HELD FIXED (the earlier sweep tied L3 to L1)")
hdg=0.15
for w3 in [0.02,0.05,0.10]:
    print(f"\n  --- wL3 = {w3:.2f} (hedgers 15%). cell = sS% / best corner ---")
    w2s=np.round(np.arange(0.00,0.361,0.04),3)
    print("        wL2:"+"".join(f"{w:>11.2f}" for w in w2s))
    for w0 in np.round(np.arange(0.15,0.661,0.05),3):
        row=f"  wL0={w0:4.2f}   "
        for w2 in w2s:
            w1=1-w0-w2-w3
            if w1<0: row+=f"{'--':>11s}"; continue
            s=np.array([w0,w1,w2,w3])@LK[:4]; s=(1-hdg)*s+hdg/3.0
            s=(1-MYW)*s+MYW*np.array([0,0,1])
            v=MY*multipliers(s); row+=f"{s[2]*100:8.1f}/{names[v.argmax()]}"
        print(row)

print("\n  Threshold on sP for Scissors, at the blend's typical sR,sS:")
for (sR,sS) in [(0.32,0.35),(0.35,0.35),(0.38,0.38),(0.40,0.35),(0.30,0.33)]:
    lo,hi=0.0,1-sR-sS
    for _ in range(60):
        mid=(lo+hi)/2; s=np.array([sR,mid,sS]); s=s/s.sum()
        v=MY*multipliers(s)
        if v.argmax()==2: hi=mid
        else: lo=mid
    print(f"    sR={sR:.2f} sS={sS:.2f} -> Scissors needs sP >~ {hi:.4f} (after renormalising)")

print("\n"+"="*78); print("GAME 4: all 81 combinations")
BID=np.array([5.,15.,25.,35.]); OFF=np.array([10.,20.,30.,40.])
def g4(sS_pct, acts):
    """acts: array of 4 in {+1 buy, -1 sell, 0 leave}"""
    a=np.asarray(acts,float)
    prof=np.where(a>0, sS_pct[:,None]-OFF, np.where(a<0, BID-sS_pct[:,None], 0.0)).sum(axis=1)
    return prof
def maxprof(sS_pct):
    per=np.maximum.reduce([np.zeros((len(sS_pct),4)), sS_pct[:,None]-OFF, BID-sS_pct[:,None]])
    return per.sum(axis=1)
for key in ["BLEND","A_user_levelk","C_ch"]:
    sS=s_with_me(D[key],[0,0,1])[:,2]*100
    mp=maxprof(sS)
    assert (mp>0).all(), "max available profit must be >0 for every draw"
    rows=[]
    for acts in itertools.product([1,-1,0],repeat=4):
        p=g4(sS,acts); sc=np.clip(p/mp,0,1)*100
        rows.append((sc.mean(),acts,np.mean(p>0)*100,np.median(sc)))
    rows.sort(reverse=True)
    sym={1:'B',-1:'S',0:'L'}
    print(f"\n  -- {key}  (mean max-available profit {mp.mean():.2f}, mean sS {sS.mean():.2f})")
    print("     rank  M1 M2 M3 M4   E[score]  P(profit>0)  median")
    for i,(m,a,pp,md) in enumerate(rows[:10]):
        print(f"      {i+1:2d}    {'  '.join(sym[x] for x in a)}    {m:7.3f}    {pp:6.2f}%   {md:6.2f}")
    print("     ... worst 3:")
    for i,(m,a,pp,md) in enumerate(rows[-3:]):
        print(f"      {len(rows)-2+i:2d}    {'  '.join(sym[x] for x in a)}    {m:7.3f}    {pp:6.2f}%   {md:6.2f}")
    # the all-Leave benchmark
    for m,a,pp,md in rows:
        if a==(0,0,0,0): print(f"     ALL-LEAVE benchmark: E[score]={m:.3f}  (rank {rows.index((m,a,pp,md))+1}/81)")

print("\n"+"="*78); print("GAME 4 SCORING EDGE CASE")
print("  score = clamp(total_profit / max_available_profit, 0, 1)*100.")
print("  A losing book scores 0. LEAVING EVERYTHING also scores 0 (0/max = 0).")
print("  => taking positions has ZERO downside relative to abstaining. 'Leave' is weakly")
print("     dominated on every market that does not drag the TOTAL below zero.")
sS=s_with_me(D['BLEND'],[0,0,1])[:,2]*100
print(f"  P(sS>10)={np.mean(sS>10)*100:.2f}%  P(sS>20)={np.mean(sS>20)*100:.2f}%  P(sS>30)={np.mean(sS>30)*100:.2f}%")
print(f"  P(sS<35)={np.mean(sS<35)*100:.2f}%  P(sS>40)={np.mean(sS>40)*100:.2f}%  P(35<=sS<=40)={np.mean((sS>=35)&(sS<=40))*100:.2f}%")
