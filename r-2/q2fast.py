import numpy as np, pickle
rng=np.random.default_rng(5)
A=np.load("mc_out.npy")[:,2]*100; N=467
def opp_quotes(n,rs):
    r=rs.random(n); B=np.empty(n); O=np.empty(n)
    d=r<0.30; B[d]=34.; O[d]=44.
    l=(r>=0.30)&(r<0.75)
    m=rs.normal(37,1.8,n); sp=rs.integers(10,15,n)
    B[l]=np.round(m-sp/2)[l]; O[l]=np.round(m+sp/2)[l]
    h=r>=0.75
    m2=rs.normal(36.5,3.5,n); sp2=rs.integers(15,31,n)
    B[h]=np.round(m2-sp2/2)[h]; O[h]=np.round(m2+sp2/2)[h]
    return B,O
def pnl_vec(myB,myO,B,O,a,conv):
    """myB,myO arrays (K,). Returns (K,) pnl."""
    Os=np.sort(O); Oc=np.cumsum(Os); Bs=np.sort(B)[::-1]; Bc=np.cumsum(Bs)
    nb=np.searchsorted(Os,myB,side='right'); ns=np.searchsorted(-Bs,-myO,side='right')
    sO=np.where(nb>0,Oc[np.clip(nb-1,0,None)],0.0); sB=np.where(ns>0,Bc[np.clip(ns-1,0,None)],0.0)
    if conv=='mine':   return nb*(a-myB)+ns*(myO-a)
    if conv=='theirs': return (nb*a-sO)+(sB-ns*a)
    return (nb*a-(nb*myB+sO)/2) + ((ns*myO+sB)/2-ns*a)
grid=np.array([(b,b+s) for b in range(28,46) for s in range(10,31)],float)
gB,gO=grid[:,0],grid[:,1]
S=6000; tot={c:np.zeros(len(grid)) for c in ['mine','theirs','mid']}
for _ in range(S):
    a=A[rng.integers(len(A))]; B,O=opp_quotes(N-1,rng)
    for c in tot:
        mp=pnl_vec(gB,gO,B,O,a,c); op=pnl_vec(B,O,B,O,a,c)
        ops=np.sort(op)
        lo=np.searchsorted(ops,mp,side='left');hi=np.searchsorted(ops,mp,side='right');tot[c]+=(lo+hi)/2/len(ops)*100
best={}
for c in tot:
    tot[c]/=S; k=np.argsort(-tot[c])[:6]
    print("Q2 %-7s top: %s"%(c,[((int(gB[i]),int(gO[i])),round(tot[c][i],1)) for i in k]))
    di=np.where((gB==34)&(gO==44))[0][0]
    print("        default(34,44)=%.1f  research(33,45)=%.1f"%(tot[c][di],tot[c][np.where((gB==33)&(gO==45))[0][0]]))
comb=sum(tot.values())/3
k=np.argsort(-comb)[:10]
print("\nQ2 ROBUST avg-over-conventions top10: %s"%[((int(gB[i]),int(gO[i])),round(comb[i],1)) for i in k])
mn=np.min(np.stack([tot[c] for c in tot]),0); km=np.argsort(-mn)[:6]
print("Q2 MAXIMIN over conventions: %s"%[((int(gB[i]),int(gO[i])),round(mn[i],1)) for i in km])
pickle.dump((grid,tot,comb,mn),open("q2res.pkl","wb"))

print("\n--- Q2 diagnostics: bid sweep, offer fixed dead ---")
for b in range(33,44):
    row=[]
    for c in ['mine','theirs','mid']:
        i=np.where((gB==b)&(gO==b+10))[0][0]; row.append(round(tot[c][i],1))
    print("   bid %d / offer %d : mine %.1f theirs %.1f mid %.1f"%(b,b+10,*row))
