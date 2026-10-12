import numpy as np
from field_core import *
np.set_printoptions(precision=4, suppress=True)
def pc(s): return np.round(np.asarray(s)*100, 2)

print("="*70); print("V1. FORMULA CHECK against the worked example")
u = np.array([1/3,1/3,1/3])
v = values(TABLES[MY_TABLE], u)
print(" table (3,5,9), uniform shares -> values", np.round(v,4), "(expect 1.7143 2.8571 5.1429)")
w = np.array([0.5,0.5,0.0])
print(" 50/50 Rock-Paper score =", round(game1_score(w, TABLES[MY_TABLE], u),4), "(expect 44.4444)")
assert np.allclose(v, [1.714286,2.857143,5.142857], atol=1e-5)
assert abs(game1_score(w,TABLES[MY_TABLE],u)-44.4444)<1e-3
print(" PASS - formula reproduces their example exactly.")

print("="*70); print("V2. SHARE-SUM AND SIGN SANITY")
print(" table fractions sum:", TFRAC.sum())

print("="*70); print("V3. LEVEL-K LADDER")
profs, mvs = level_profiles(6)
names="RPS"
for k in range(7):
    print(f"  L{k}: sR/sP/sS = {pc(profs[k])}   moves by table {''.join(names[m] for m in mvs[k])}  sum={profs[k].sum():.6f}")
print("\n  L1 multipliers from L0 shares:", np.round(multipliers(profs[0]),6), "(expect 1.022222 0.338983 0.478873)")
v1 = values(TABLES, profs[0])
print("  per-table values at L0 shares:")
for t in range(6):
    print(f"    T{t} {TABLES[t].astype(int)} frac {TFRAC[t]:.2f}: R={v1[t,0]:.4f} P={v1[t,1]:.4f} S={v1[t,2]:.4f} -> {names[v1[t].argmax()]}  margin={np.sort(v1[t])[-1]-np.sort(v1[t])[-2]:.4f}")

print("="*70); print("V4. IS THERE A PURE FIXED POINT? / ORBIT")
seen={}
for k in range(7):
    key=tuple(np.round(profs[k],9))
    if key in seen: print(f"  cycle: L{k} == L{seen[key]}  -> period {k-seen[key]}"); break
    seen[key]=k
for k in range(6):
    fixed = np.allclose(profs[k], profs[k+1])
    if fixed: print(f"  L{k} IS a fixed point")
print("  no pure fixed point found among L0..L6" if not any(np.allclose(profs[k],profs[k+1]) for k in range(6)) else "")

print("="*70); print("V5. COGNITIVE HIERARCHY, tau=1.5")
agg, chp, chm, w = ch_aggregate(1.5, kmax=8)
for k in range(6):
    print(f"  CH-L{k}: sR/sP/sS = {pc(chp[k])}  moves {''.join(names[m] for m in chm[k])}")
print("  Poisson weights (tail folded into last):", np.round(w,4))
print("  CH population aggregate  sR/sP/sS =", pc(agg), " sum=", agg.sum())
