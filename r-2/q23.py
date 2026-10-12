import numpy as np
rng=np.random.default_rng(3)
out=np.load("mc_out.npy"); A=out[:,2]*100; T=len(A); N=467
P=np.load("bucketP.npy")
EDGE=[-1e9,30,34,38,41,1e9]
def bidx(a): return np.searchsorted([30,34,38,41],a,side='right')

# ================= Q3 : RANK simulation =================
# opponent entry model: LLM bloc piles on modal bucket; humans spread.
def opp_entries(n,rs):
    E=np.zeros((n,5))
    r=rs.random(n)
    llm=0.40
    # LLM bloc: all-in or near-all-in on bucket 3 (34-38) -- the research-pass modal answer
    k=int(n*llm)
    E[:k,2]=82; E[:k,1]=18                      # exactly the research pass recommendation
    j=int(n*llm*0.35)
    E[:j,:]=0; E[:j,2]=100                      # a sub-bloc goes pure modal
    # humans: dirichlet around a smeared prior
    m=n-k
    base=np.array([3,15,45,27,10],float)
    E[k:]=rs.dirichlet(base/3.0,m)*100
    return E
def dev(E,tb): return 2*(100-E[:,tb])
CAND=[]
for b3 in range(0,101,5):
  for b4 in range(0,101-b3,5):
    for b2 in range(0,101-b3-b4,5):
      b5=100-b3-b4-b2
      if b5>60: continue
      CAND.append((0,b2,b3,b4,b5))
CAND=[np.array(c) for c in CAND]
R=3000
tb=rng.choice(5,R,p=P/P.sum())
res=np.zeros(len(CAND))
for r in range(R):
    E=opp_entries(N-1,rng); od=dev(E,tb[r])
    for i,c in enumerate(CAND):
        d=2*(100-c[tb[r]])
        res[i]+= (od>d).mean()*100 + (od==d).mean()*50
res/=R
o=np.argsort(-res)
print("Q3 RANK-SCORE SIM (R=%d, 467 opponents, 40%% LLM bloc on 34-38)"%R)
for i in o[:8]: print("   %s  E[percentile]=%.2f"%(CAND[i],res[i]))
print("   pure 34-38 (0,0,100,0,0): %.2f"%res[[i for i,c in enumerate(CAND) if tuple(c)==(0,0,100,0,0)][0]])
print("   pure 38-41 (0,0,0,100,0): %.2f"%res[[i for i,c in enumerate(CAND) if tuple(c)==(0,0,0,100,0)][0]])
print("   research  (0,18,82,0,0): see grid")
np.save("q3res.npy",np.array([res,[0]*len(res)]))
import pickle; pickle.dump((CAND,res),open("q3.pkl","wb"))

# ================= Q2 : market making =================
def opp_quotes(n,rs):
    r=rs.random(n); B=np.empty(n); O=np.empty(n)
    d=r<0.30                                   # default keepers 34/44
    B[d]=34; O[d]=44
    l=(r>=0.30)&(r<0.75)                       # LLM-advised: centred ~36-38, spread 10-14
    m=rs.normal(37,1.8,n); sp=rs.integers(10,15,n)
    B[l]=np.round(m-sp/2)[l]; O[l]=np.round(m+sp/2)[l]
    h=r>=0.75                                  # humans: wide, anchored on Day-1
    m2=rs.normal(36.5,3.5,n); sp2=rs.integers(15,31,n)
    B[h]=np.round(m2-sp2/2)[h]; O[h]=np.round(m2+sp2/2)[h]
    return B,O
def pnl(myB,myO,B,O,a,conv):
    buy=O<=myB; sell=B>=myO
    if conv=='mine':   pb=myB; ps=myO
    elif conv=='theirs': pb=O[buy]; ps=B[sell]
    else: pb=(myB+O[buy])/2; ps=(myO+B[sell])/2
    return np.sum(a-pb if np.ndim(pb) else buy.sum()*(a-pb)) + np.sum(ps-a if np.ndim(ps) else sell.sum()*(ps-a))
S=1200
grid=[(b,b+s) for b in range(20,50) for s in range(10,31)]
tot={c:np.zeros(len(grid)) for c in ['mine','theirs','mid']}
for s_ in range(S):
    a=A[rng.integers(T)]; B,O=opp_quotes(N-1,rng)
    for c in tot:
        mypnl=np.array([pnl(b,o,B,O,a,c) for b,o in grid])
        opp=np.array([pnl(B[i],O[i],np.delete(B,i),np.delete(O,i),a,c) for i in rng.integers(0,N-1,60)])
        tot[c]+=np.array([ (opp<mp).mean()*100 for mp in mypnl])
for c in tot:
    tot[c]/=S; k=np.argsort(-tot[c])[:5]
    print("\nQ2 conv=%-7s top: %s"%(c,[(grid[i],round(tot[c][i],1)) for i in k]))
    print("      default 34/44 -> %.1f"%tot[c][grid.index((34,44))])
comb=sum(tot.values())/3; k=np.argsort(-comb)[:8]
print("\nQ2 ROBUST (avg over 3 conventions) top: %s"%[(grid[i],round(comb[i],1)) for i in k])
np.save("q2.npy",np.array([comb,[g[0] for g in grid],[g[1] for g in grid]]))
import pickle; pickle.dump((grid,tot,comb),open("q2.pkl","wb"))
