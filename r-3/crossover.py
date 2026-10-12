import numpy as np
from core import *
# Exact S-vs-P crossover for my table: 9 sP/(sS+.25) = 5 sR/(sP+.25), with sR = 1-sP-sS
print("MY TABLE (3,5,9): Paper beats Scissors when s_P falls below this, for given s_S:")
for sS in [0.40,0.42,0.43,0.44,0.45,0.46]:
    lo=None
    for sP in np.arange(0.40,0.05,-0.0005):
        sR=1-sS-sP
        if sR<0: continue
        v=values(MINE,(sR,sP,sS))
        if v[1]>v[2]: lo=sP; break
    # closed form: 9 sP (sP+.25) = 5 (1-sS-sP)(sS+.25)  -> quadratic
    a=9; b=9*0.25+5*(sS+0.25); c=-5*(1-sS)*(sS+0.25)
    x=(-b+np.sqrt(b*b-4*a*c))/(2*a)
    print("   s_S=%.2f : crossover s_P=%.4f (s_R=%.4f)  closed-form %.4f"%(sS,lo,1-sS-lo,x))
print("\nRock beats Scissors when s_S above (for given s_P):")
for sP in [0.20,0.25,0.30,0.34]:
    a=1;b=0.25+3*sP;c=-3*sP*(1-sP+0.25)   # 3 sP (sR+.25) = sS(sS+.25), sR=1-sP-sS
    # sS^2+.25 sS = 3 sP (1.25 - sP - sS) -> sS^2 + (.25+3sP) sS - 3sP(1.25-sP)=0
    x=(-b+np.sqrt(b*b-4*a*c))/2
    print("   s_P=%.2f : Rock wins if s_S > %.4f"%(sP,x))
# Scenario table: Paper->Rock shift of x points from Day-2 L1 projection
print("\nSCENARIO: S=43.5 fixed; x points of field move Paper->Rock from baseline R22.5/P34.0")
for x in [0,2,4,6,8,10,12,15,20]:
    s=((22.5+x)/100,(34-x)/100,0.435); v=values(MINE,s)
    print("   shift %2d -> R%.1f P%.1f S%.1f : vR %.2f vP %.2f vS %.2f  best=%s  S gets %.0f%%  P gets %.0f%%"%(
        x,*np.array(s)*100,*v,"RPS"[v.argmax()],100*v[2]/v.max(),100*v[1]/v.max()))
