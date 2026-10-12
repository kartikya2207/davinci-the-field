import numpy as np
rng=np.random.default_rng(21); N=467
# BLEND: my MC (38.4) 55% + research-pass model (37.0) 45%  -> honest model uncertainty
Pmine=np.array([0.000,0.014,0.412,0.485,0.089]); Pres=np.array([0.05,0.18,0.52,0.17,0.08])
P=0.55*Pmine+0.45*Pres; P/=P.sum()
print("BLENDED bucket P = %s"%np.round(P*100,2))
def opp(n,rs,mode):
    E=np.zeros((n,5)); k=int(n*0.42)
    if mode=='res': E[:k]= [0,18,82,0,0]
    else:           E[:k]= [0,0,100,0,0]
    j=int(k*0.4); E[:j]=0; E[:j,2]=100
    E[k:]=rs.dirichlet(np.array([2,10,40,28,8])/3.,n-k)*100
    return E
CAND=[]
for b2 in range(0,26,5):
 for b3 in range(0,101,5):
  for b4 in range(0,101-b3,5):
   b5=100-b2-b3-b4
   if 0<=b5<=30: CAND.append(np.array([0,b2,b3,b4,b5]))
R=2500; res=np.zeros(len(CAND)); C=np.array(CAND)
tb=rng.choice(5,R,p=P)
for r in range(R):
    E=opp(N-1,rng,'res' if r%2 else 'pure'); od=2*(100-E[:,tb[r]])
    d=2*(100-C[:,tb[r]])
    res+= (od[None,:]>d[:,None]).mean(1)*100+(od[None,:]==d[:,None]).mean(1)*50
res/=R; o=np.argsort(-res)
print("\nQ3 FINAL RANK SIM  (467 opponents, 42%% LLM bloc)")
for i in o[:10]: print("   %s  E[pct]=%.2f"%(C[i],res[i]))
for t in [(0,0,100,0,0),(0,0,0,100,0),(0,18,82,0,0),(0,0,55,40,5),(0,5,50,40,5),(0,0,50,45,5)]:
    m=np.where((C==np.array(t)).all(1))[0]
    if len(m): print("   ref %s -> %.2f"%(str(t),res[m[0]]))
# raw-deviation check
print("\n   raw E[deviation]: pure-b3=%.1f pure-b4=%.1f best-rank=%.1f"%(
   200-2*100*P[2],200-2*100*P[3],200-2*(C[o[0]]*P).sum()))
