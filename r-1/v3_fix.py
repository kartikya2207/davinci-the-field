import numpy as np
from scipy.optimize import minimize
from field_core import *
np.set_printoptions(precision=4, suppress=True)
def pc(s): return np.round(np.asarray(s)*100,2)
names="RPS"; MYW=1/200.0
MY=TABLES[MY_TABLE]

n=30; G=np.array([[i/n,j/n,(n-i-j)/n] for i in range(n+1) for j in range(n+1-i)])
CORN=np.eye(3)
def scores_for(s_oth,W):
    s=(1-MYW)*s_oth[:,None,:]+MYW*W[None,:,:]
    v=MY*multipliers(s)
    return 100.0*np.sum(W[None,:,:]*v,axis=2)/v.max(axis=2)

print("="*70); print("V6b. Does the 0.5% self-effect ever break the corner result? (MY table 3/5/9)")
rng=np.random.default_rng(11)
for label,mode in [("uniform Dirichlet(1,1,1) [stress test]","u"),
                   ("plausible box: every share in 0.15-0.55","p")]:
    if mode=="u": S=rng.dirichlet([1,1,1],40000)
    else:
        S=rng.dirichlet([1,1,1],600000); S=S[np.all((S>0.15)&(S<0.55),axis=1)][:40000]
    nb=0; worst=0.0; wk=None; mn=1e9; over=False
    for a in range(0,len(S),2000):
        ch=S[a:a+2000]
        cs=scores_for(ch,CORN); gs=scores_for(ch,G)
        bc=cs.max(axis=1); bg=np.maximum(gs.max(axis=1),bc)
        g=bg-bc; nb+=int((g>1e-9).sum()); mn=min(mn,bc.min())
        over|=bool(gs.max()>100+1e-9)
        if g.max()>worst: worst=g.max(); wk=(ch[g.argmax()].copy(), cs[g.argmax()].copy())
    print(f"  {label}: N={len(S)}  mix>best corner in {nb} ({100*nb/len(S):.4f}%)  worst gain {worst:.4f} pts")
    print(f"     min best-corner score {mn:.3f};  any score >100? {over}")
    if wk is not None and worst>0: print(f"     worst at s_others={pc(wk[0])} corner scores={np.round(wk[1],3)}")

print("="*70); print("V8b. TRUE MIXED EQUILIBRIUM via regret minimisation")
def unpack(z):
    y=z.reshape(6,3); y=y-y.max(axis=1,keepdims=True); x=np.exp(y); return x/x.sum(axis=1,keepdims=True)
def regret(z):
    x=unpack(z); s=(TFRAC[:,None]*x).sum(axis=0); v=values(TABLES,s)
    return float(np.sum(TFRAC*(v.max(axis=1)-np.sum(x*v,axis=1))))
best=None; rng=np.random.default_rng(3)
for _ in range(60):
    r=minimize(regret,rng.normal(0,2,18),method='Powell',options={'maxiter':40000,'xtol':1e-11,'ftol':1e-13})
    if best is None or r.fun<best.fun: best=r
best=minimize(regret,best.x,method='Nelder-Mead',options={'maxiter':60000,'fatol':1e-15,'xatol':1e-13})
x=unpack(best.x); s_eq=(TFRAC[:,None]*x).sum(axis=0); v=values(TABLES,s_eq)
print(f"  min total regret = {best.fun:.3e}   (0 => exact equilibrium)")
print("  equilibrium sR/sP/sS =",pc(s_eq)," sum=",round(float(s_eq.sum()),9))
for t in range(6):
    srt=np.sort(v[t])[::-1]
    print(f"    T{t} {TABLES[t].astype(int)} f={TFRAC[t]:.2f}: x={np.round(x[t],4)} v={np.round(v[t],4)} gap={srt[0]-srt[1]:.5f}")
print("  MY table:", "MIXED" if x[MY_TABLE].max()<0.999 else "PURE", np.round(x[MY_TABLE],4))
print("  MY argmax:",names[v[MY_TABLE].argmax()]," score all-in S:",round(game1_score([0,0,1],MY,s_eq),2),
      " all-in P:",round(game1_score([0,1,0],MY,s_eq),2)," all-in R:",round(game1_score([1,0,0],MY,s_eq),2))
np.save("s_eq.npy",s_eq)
