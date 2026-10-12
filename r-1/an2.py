import numpy as np
from field_core import *
from mc import s_with_me, MYW, simulate
np.set_printoptions(precision=3, suppress=True)
names="RPS"; MY=TABLES[MY_TABLE]
EDGES=[30.,34.,38.,41.]; BNAME=["<30","30-34","34-38","38-41","41+"]
LK,_=level_profiles(6)

print("="*78); print("GAME 1: THE EXACT FLIP BOUNDARY FOR MY TABLE (3,5,9)")
print("  S beats P  <=>  9*sP/(sS+.25) > 5*sR/(sP+.25)  <=>  9*sP*(sP+.25) > 5*sR*(sS+.25)")
print("  S beats R  <=>  9*sP/(sS+.25) > 3*sS/(sR+.25)  <=>  9*sP*(sR+.25) > 3*sS*(sS+.25)")
print("\n  Critical sP at which S ties P (solving 9 sP^2 + 2.25 sP - 5 sR(sS+.25) = 0):")
print("   sR     sS    sP*_SvP   sP*_SvR   sP* needed   (sP available = 1-sR-sS)   S wins?")
for sR in [0.20,0.25,0.30,0.35,0.38,0.40,0.45,0.50]:
    for sS in [0.30,0.35,0.40]:
        c=5*sR*(sS+.25); sp_p=(-2.25+np.sqrt(2.25**2+4*9*c))/(2*9)
        sp_r=3*sS*(sS+.25)/(9*(sR+.25))
        need=max(sp_p,sp_r); avail=1-sR-sS
        print(f"  {sR:5.2f} {sS:6.2f}   {sp_p:7.4f}   {sp_r:7.4f}    {need:7.4f}        {avail:7.4f}          {'YES' if avail>need else 'no'}")
print("\n  AUDIT of the report's claim 'flip at sR > 0.38':")
for sR in [0.34,0.36,0.373,0.38,0.40,0.42]:
    sS=sR; sP=1-2*sR
    v=MY*multipliers(np.array([sR,sP,sS]))
    print(f"   along sR=sS: sR={sR:.3f} sP={sP:.3f} -> v = {np.round(v,4)} best={names[v.argmax()]}")
print("   -> along the sR=sS diagonal the flip is near sR~0.373, so '0.38' is COINCIDENTALLY close,")
print("      but it is NOT an sR threshold: raise sP and S wins at any sR. Counter-example:")
for s in [[0.45,0.25,0.30],[0.50,0.30,0.20],[0.38,0.12,0.50],[0.38,0.32,0.30]]:
    v=MY*multipliers(np.array(s)); print(f"      s={np.round(np.array(s)*100,1)} -> {np.round(v,3)} best={names[v.argmax()]}")

print("\n"+"="*78); print("GAME 1: how low must sP fall before Scissors stops winning?")
D=np.load("mc_out.npz")
for k in ["BLEND"]:
    s=D[k]; sm=s_with_me(s,[0,0,1]); v=MY*multipliers(sm); am=v.argmax(axis=1)
    for m,nm in enumerate(names):
        sel=am==m
        if sel.sum()>0:
            print(f"   when best corner is {nm} ({100*sel.mean():5.2f}% of trials): mean sR/sP/sS = {np.round(sm[sel].mean(axis=0)*100,2)}"
                  f"  sP p95={np.percentile(sm[sel,1],95)*100:5.2f}")
    sP=sm[:,1]*100
    for thr in [10,15,20,22,25,28,30]:
        sel=np.abs(sP-thr)<1.0
        if sel.sum()>50:
            print(f"   sP ~ {thr}%: P(Scissors is best corner) = {100*np.mean(am[sel]==2):5.1f}%   (n={sel.sum()})")

print("\n"+"="*78); print("SENSITIVITY SWEEP  (deterministic, hedgers=15%, L1:L3 split the remainder 8:1)")
print("  cell = sS%% including me all-in S | bucket")
w0s=np.round(np.arange(0.10,0.701,0.05),3); w2s=np.round(np.arange(0.05,0.401,0.05),3)
hdg=0.15
print("        wL2:"+"".join(f"{w:>13.2f}" for w in w2s))
for w0 in w0s:
    row=f"  wL0={w0:4.2f}   "
    for w2 in w2s:
        rem=1-w0-w2
        if rem<0: row+=f"{'--':>13s}"; continue
        w=np.array([w0, rem*8/9, w2, rem/9])
        s=w@LK[:4]; s=(1-hdg)*s+hdg/3.0; s=(1-MYW)*s+MYW*np.array([0,0,1])
        sS=s[2]*100; b=BNAME[int(np.searchsorted(EDGES,sS,side='right'))]
        v=MY*multipliers(s)
        row+=f"{sS:8.1f}/{names[v.argmax()]}{'*' if b in('30-34','34-38') else ' '}"
    print(row)
print("  (letter = my best Game-1 corner at that point; * marks the two contested buckets)")

print("\n  Bucket at each sweep point:")
print("        wL2:"+"".join(f"{w:>9.2f}" for w in w2s))
for w0 in w0s:
    row=f"  wL0={w0:4.2f}   "
    for w2 in w2s:
        rem=1-w0-w2
        if rem<0: row+=f"{'--':>9s}"; continue
        w=np.array([w0, rem*8/9, w2, rem/9])
        s=w@LK[:4]; s=(1-hdg)*s+hdg/3.0; s=(1-MYW)*s+MYW*np.array([0,0,1])
        row+=f"{BNAME[int(np.searchsorted(EDGES,s[2]*100,side='right'))]:>9s}"
    print(row)
