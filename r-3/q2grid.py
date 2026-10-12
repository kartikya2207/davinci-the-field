"""JOB 1: Q2 full grid bids 34-42 x spread 10-14, centred 42.4 (ensemble) and 43.5 (recruitment fit), bull/bear sweep, 3 conventions."""
import numpy as np, sys
exec(open("q2v2.py").read().split("grid=np.array")[0])
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
E=np.load("ensemble_out.npy")[:,2]*100
grid=np.array([(b,b+s) for b in range(34,43) for s in range(10,15)],float); gB,gO=grid[:,0],grid[:,1]
scen=[(bu,be) for bu in [0.02,0.05,0.08,0.12] for be in [0.0,0.02,0.04,0.06]]
for shift,lab in [(0.0,"centre 42.4"),(1.1,"centre 43.5")]:
    A=E+shift; res={}
    for bu,be in scen:
        tot={c:np.zeros(len(grid)) for c in ['mine','theirs','mid']}
        for _ in range(400):
            a=A[rng.integers(len(A))]; c3=rng.uniform(42,45)+shift*0.5; B,O=field(517,rng,bu,be,c3)
            for c in tot: tot[c]+=pct(gB,gO,B,O,a,c)
        for c in tot: tot[c]/=400
        res[(bu,be)]=tot
    avg=np.mean([sum(t.values())/3 for t in res.values()],0); mn=np.min([np.min(np.stack(list(t.values())),0) for t in res.values()],0)
    print("\n=== %s (mean S=%.1f, P(S>=45)=%.0f%%, P(S>=44)=%.0f%%) ===  E[pct] avg over 16 bull/bear scen x 3 conv | maximin"%(lab,A.mean(),100*(A>=45).mean(),100*(A>=44).mean()))
    print("bid\spread  "+"  ".join("%5d"%s for s in range(10,15))+"   |  maximin at spread 10..14")
    for b in range(34,43):
        row=[avg[np.where((gB==b)&(gO==b+s))[0][0]] for s in range(10,15)]; rm=[mn[np.where((gB==b)&(gO==b+s))[0][0]] for s in range(10,15)]
        print("  %2d        "%b+"  ".join("%5.1f"%x for x in row)+"   |  "+"  ".join("%5.1f"%x for x in rm))
    for c in ['mine','theirs','mid']:
        v=np.mean([t[c] for t in res.values()],0); k=np.argsort(-v)[:4]; print("   %-6s best: %s"%(c,[((int(gB[i]),int(gO[i])),round(v[i],1)) for i in k]))
    print("   by scenario (bull,bear) -> best quote:")
    for (bu,be),t in res.items():
        v=sum(t.values())/3; i=v.argmax(); print("      bull %2.0f%% bear %.0f%%: best %d/%d (%.1f)  | 35/45=%.1f 37/47=%.1f 39/49=%.1f 40/50=%.1f"%(bu*100,be*100,gB[i],gO[i],v[i],*[v[np.where((gB==b)&(gO==b+10))[0][0]] for b in (35,37,39,40)]))
