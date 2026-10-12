"""r-2 core. Self-contained. Nothing imported from r-1."""
import numpy as np

TABLES = {  # name -> (payR,payP,payS), share of field
 (3,5,9):0.34,(5,9,3):0.24,(9,3,5):0.14,(5,3,9):0.12,(3,9,5):0.10,(9,5,3):0.06}
MINE=(3,5,9)

def values(pay,s):
    """s=(sR,sP,sS) fractions. Returns (vR,vP,vS)."""
    pR,pP,pS=pay; sR,sP,sS=s
    return (pR*sS/(sR+0.25), pP*sR/(sP+0.25), pS*sP/(sS+0.25))

def score_q1(pay,w,s):
    """w = weights summing to 1 (order R,P,S). Returns 0-100."""
    v=values(pay,s); return 100.0*sum(wi*vi for wi,vi in zip(w,v))/max(v)

# ---- VERIFY against official worked example ----
if __name__=="__main__":
    s=(1/3,1/3,1/3); v=values((3,5,9),s)
    print("equal thirds (3,5,9) ->",[round(x,3) for x in v], "expect 1.714 2.857 5.143")
    avg=0.5*v[0]+0.5*v[1]
    print("50/50 R/P avg=%.3f expect 2.286 ; score=%.1f expect 44.4"%(avg,100*avg/max(v)))
    assert abs(v[0]-1.714)<0.002 and abs(v[1]-2.857)<0.002 and abs(v[2]-5.143)<0.002
    assert abs(100*avg/max(v)-44.4)<0.1
    print("FORMULA VERIFIED OK")
    # BR of every table to day-1 aggregate
    d1=(0.307,0.325,0.369)
    for t in TABLES:
        v=values(t,d1); print(t,[round(x,3) for x in v],"BR=","RPS"[int(np.argmax(v))])
