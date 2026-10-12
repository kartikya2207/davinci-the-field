"""r-3 core. Self-contained; nothing imported from r-1 or r-2."""
import numpy as np
TABLES={(3,5,9):0.34,(5,9,3):0.24,(9,3,5):0.14,(5,3,9):0.12,(3,9,5):0.10,(9,5,3):0.06}
MINE=(3,5,9); N3=518
D1=np.array([0.307,0.325,0.369]); D2=np.array([0.257,0.329,0.413])
UNI=np.array([1/3,1/3,1/3]); L1=np.array([0.20,0.34,0.46])
BUCKET_EDGES=[30,34,38,41]           # <30 | 30-34 | 34-38 | 38-41 | >=41
MARKETS=[(5,10),(15,20),(25,30),(35,40)]

def values(pay,s):
    pR,pP,pS=pay; sR,sP,sS=s
    return np.array([pR*sS/(sR+0.25), pP*sR/(sP+0.25), pS*sP/(sS+0.25)])
def score_q1(pay,w,s):
    v=values(pay,s); w=np.asarray(w,float)/np.sum(w); return 100*(w@v)/v.max()
def bucket(sS_pct):
    return np.searchsorted(BUCKET_EDGES, sS_pct, side='right')   # 0..4
def q3_dev(alloc,true_bucket): return 2*(100-alloc[true_bucket])
def q4_profit(dec,a):
    """dec: 4-tuple in {1 buy,-1 sell,0 leave}; a: actual % (scalar or array)."""
    p=np.zeros_like(np.asarray(a,float))
    for d,(b,o) in zip(dec,MARKETS):
        if d==1: p=p+(a-o)
        elif d==-1: p=p+(b-a)
    return p
def q4_avail(a):
    a=np.asarray(a,float); av=np.zeros_like(a)
    for b,o in MARKETS: av=av+np.maximum(np.maximum(a-o,b-a),0)
    return av
def q4_score(dec,a):
    av=q4_avail(a); return np.clip(100*q4_profit(dec,a)/np.where(av>0,av,1e-9),0,100)

if __name__=="__main__":
    v=values(MINE,UNI); print("worked ex:",np.round(v,3),"expect 1.714 2.857 5.143")
    assert np.allclose(v,[1.714,2.857,5.143],atol=2e-3)
    print("50/50 R/P score:",round(score_q1(MINE,[50,50,0],UNI),1),"expect 44.4")
    assert abs(score_q1(MINE,[50,50,0],UNI)-44.4)<0.1
    # Day-2 ground truth Q1
    v2=values(MINE,D2); print("Day2 values:",np.round(v2,3),"expect 2.444 2.219 4.466")
    assert np.allclose(v2,[2.444,2.219,4.466],atol=2e-3)
    print("Day2 Q1 score pure S:",score_q1(MINE,[0,0,100],D2),"(actual 100.0)")
    # Day-2 Q3 ground truth: alloc (0,0,55,40,5) at 41.3 -> bucket 4 -> dev 190
    print("Day2 bucket of 41.3 =",bucket(41.3),"dev=",q3_dev([0,0,55,40,5],bucket(41.3)),"expect 4, 190")
    print("Day1 bucket of 36.9 =",bucket(36.9),"dev=",q3_dev([0,0,100,0,0],bucket(36.9)),"expect 2, 0")
    # Day-2 Q4 ground truth: BBBL at 41.3
    print("Day2 Q4 BBBL score=",round(float(q4_score((1,1,1,0),41.3)),2),"(actual 98.0); BBBB=",round(float(q4_score((1,1,1,1),41.3)),2))
    print("Day1 Q4 BBBL score at 36.9 =",round(float(q4_score((1,1,1,0),36.9)),2))
    print("CORE VERIFIED")
