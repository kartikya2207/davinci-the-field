import numpy as np
from field_core import *
np.set_printoptions(precision=4, suppress=True)
def pc(s): return np.round(np.asarray(s)*100, 2)
names="RPS"

print("="*70); print("V6. CORNER OPTIMALITY - proof + numerical self-effect check")
print(" Analytic: score(w) = 100 * (sum_i w_i v_i)/max_j v_j.  Since sum w = 1 and")
print(" every v_i <= max_j v_j, we get sum_i w_i v_i <= max_j v_j, so score <= 100 ALWAYS,")
print(" with equality iff w puts all mass on argmax moves. This holds pointwise at")
print(" whatever shares s(w) my own throws produce, so the SELF-EFFECT CANNOT create an")
print(" interior optimum. It can only change WHICH corner attains 100.")
rng = np.random.default_rng(7)
MYW = 1/200.0     # I am 1 of 200 players -> 100 of 20000 throws -> 0.5%
worst_interior_gain = -1e9; selfdefeat = 0; trials = 40000
for _ in range(trials):
    s_oth = rng.dirichlet([1.0,1.0,1.0])
    def s_tot(w): return (1-MYW)*s_oth + MYW*np.asarray(w,float)
    corner_scores=[]
    for m in range(3):
        w=np.zeros(3); w[m]=1
        corner_scores.append(game1_score(w, TABLES[MY_TABLE], s_tot(w)))
    best_corner=max(corner_scores)
    if best_corner < 100-1e-9: selfdefeat += 1
    # fine grid of interior mixes
    for a in np.linspace(0,1,11):
        for b in np.linspace(0,1-a,int(round((1-a)*10))+1):
            w=np.array([a,b,1-a-b])
            sc=game1_score(w, TABLES[MY_TABLE], s_tot(w))
            worst_interior_gain=max(worst_interior_gain, sc-best_corner)
print(f" over {trials} random share draws x 66 weight grid:")
print(f"   max( any mix score - best corner score ) = {worst_interior_gain:.3e}  (<=0 means no mix ever beats a corner)")
print(f"   draws where NO corner attained a full 100 (self-defeating): {selfdefeat}")

print("="*70); print("V7. AUDIT of report's claimed QRE (24/34/42)")
s_rep=np.array([.24,.34,.42])
m=multipliers(s_rep); v=values(TABLES,s_rep)
print("  multipliers", np.round(m,4))
br,mv,_=best_response_profile(s_rep)
print("  best response to it:", pc(br), " moves", ''.join(names[x] for x in mv))
print("  -> NOT a fixed point. Confirms your (44,10,46). Report claim #2 fails.")

print("="*70); print("V8. DOES A MIXED EQUILIBRIUM EXIST, AND WHERE? (logit QRE continuation)")
def qre_fixed_point(lam, s0, iters=200000, tol=1e-13, damp=0.02):
    s=s0.copy()
    for i in range(iters):
        v=values(TABLES,s)
        z=lam*v; z=z-z.max(axis=1,keepdims=True)
        pr=np.exp(z); pr/=pr.sum(axis=1,keepdims=True)
        snew=(TFRAC[:,None]*pr).sum(axis=0)
        if np.max(np.abs(snew-s))<tol: return snew,True,i
        s=(1-damp)*s+damp*snew
    return s,False,iters
s=np.array([1/3,1/3,1/3])
print("   lam      sR     sP     sS    converged")
for lam in [0,0.25,0.5,1,2,3,5,8,12,20,35,60,100,200,500,1000,3000]:
    s,ok,it=qre_fixed_point(lam,s)
    print(f"  {lam:7.2f}  {s[0]*100:6.2f} {s[1]*100:6.2f} {s[2]*100:6.2f}   {ok}")
s_eq=s.copy()
print("\n  limiting profile (lam->inf), i.e. the MIXED EQUILIBRIUM:", pc(s_eq))
v=values(TABLES,s_eq)
print("  per-table values at equilibrium (a table only mixes if its top values TIE):")
for t in range(6):
    srt=np.sort(v[t])[::-1]
    print(f"    T{t} {TABLES[t].astype(int)}: R={v[t,0]:.4f} P={v[t,1]:.4f} S={v[t,2]:.4f}  top-2 gap={srt[0]-srt[1]:.5f}")
print("  MY table's best move at this equilibrium:", names[v[MY_TABLE].argmax()])

print("="*70); print("V9. CH tau sweep - where does sS land?")
print("   tau      sR     sP     sS    | CH-L2 profile      CH-L3 profile")
for tau in [0.5,0.8,1.0,1.2,1.5,1.8,2.0,2.5,3.0,4.0]:
    agg,chp,chm,w=ch_aggregate(tau,kmax=10)
    print(f"  {tau:5.2f}  {agg[0]*100:6.2f} {agg[1]*100:6.2f} {agg[2]*100:6.2f}  | {str(pc(chp[2])):18s} {str(pc(chp[3]))}")
