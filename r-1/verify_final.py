import numpy as np
from field_core import *
from mc import s_with_me, MYW
names="RPS"; MY=TABLES[MY_TABLE]
S_OTH=np.load("blend12.npy")
print("="*80); print("INDEPENDENT RE-DERIVATIONS (closed form vs simulation)")

print("\n1) Game 1 at the blend MEDIAN point, done by hand:")
med=np.median(s_with_me(S_OTH,[0,0,1]),axis=0); med=med/med.sum()
sR,sP,sS=med
mR=sS/(sR+.25); mP=sR/(sP+.25); mS=sP/(sS+.25)
print(f"   s=({sR*100:.2f},{sP*100:.2f},{sS*100:.2f})  m=({mR:.5f},{mP:.5f},{mS:.5f})")
print(f"   v_R=3*{mR:.5f}={3*mR:.4f}  v_P=5*{mP:.5f}={5*mP:.4f}  v_S=9*{mS:.5f}={9*mS:.4f}")
v=MY*multipliers(med)
print(f"   code gives {np.round(v,4)}  -> agree: {np.allclose(v,[3*mR,5*mP,9*mS])}")
print(f"   best={names[v.argmax()]};  all-in S scores {100*v[2]/v.max():.2f}")

print("\n2) Game 4 'B,B,B,S' hand-check at three sS values:")
BID=np.array([5.,15.,25.,35.]); OFF=np.array([10.,20.,30.,40.])
for x in [25.,35.,45.]:
    prof=(x-10)+(x-20)+(x-30)+(35-x)
    per=[max(0,x-10,5-x),max(0,x-20,15-x),max(0,x-30,25-x),max(0,x-40,35-x)]
    print(f"   sS={x:5.1f}: BBBS profit={prof:6.1f}  max available={sum(per):6.1f}  score={100*max(0,min(1,prof/sum(per))):6.2f}")
print("   (confirms the ratio effect: selling M4 pays most when the denominator is smallest)")

print("\n3) Game 1 flip threshold - solve 9*sP*(sP+.25) = 5*sR*(sS+.25) exactly")
for sR,sS in [(0.323,0.350),(0.35,0.35),(0.40,0.35)]:
    A,B,C=9.,2.25,-5*sR*(sS+.25)
    root=(-B+np.sqrt(B*B-4*A*C))/(2*A)
    chk=np.array([sR,root,sS]); chk=chk/chk.sum(); vv=MY*multipliers(chk)
    print(f"   sR={sR:.3f} sS={sS:.3f} -> sP* = {root:.4f};  at that point v_P={vv[1]:.4f} v_S={vv[2]:.4f} (should tie after renorm)")

print("\n4) Consistency: does level-k L2 really give sP=100 while CH-L2 gives 86?")
lk,mk=level_profiles(3); chp,chm,_=ch_profiles(1.5,3)
print(f"   level-k L2 moves {''.join(names[m] for m in mk[2])} -> {np.round(lk[2]*100,1)}")
print(f"   CH     L2 moves {''.join(names[m] for m in chm[2])} -> {np.round(chp[2]*100,1)}")
print(f"   the difference is table (9,3,5) [14% of field]: under CH it keeps Rock.")
bel=0.4*lk[0]+0.6*lk[1]
print(f"   CH-L2 belief = 0.4*L0+0.6*L1 = {np.round(bel*100,2)} (Poisson(1.5) weights 0.2231,0.3347 normalised)")
vv=values(TABLES[2],bel); print(f"   values for (9,3,5) at that belief: {np.round(vv,4)} -> {names[vv.argmax()]}")

print("\n5) Magnitude sanity across every reported quantity:")
sm=s_with_me(S_OTH,[0,0,1])
print(f"   shares sum to 1: {np.allclose(sm.sum(axis=1),1)}   all in [0,1]: {sm.min()>=0 and sm.max()<=1}")
g1=[]
for m in range(3):
    w=np.zeros(3); w[m]=1; s=s_with_me(S_OTH,w); v=MY*multipliers(s); g1.append(100*v[:,m]/v.max(axis=1))
g1=np.array(g1)
print(f"   Game1 scores in [0,100]: {g1.min()>=0 and g1.max()<=100+1e-9}  (min {g1.min():.3f} max {g1.max():.3f})")
print(f"   my table is {MY.astype(int)} = (R3,P5,S9): {list(MY.astype(int))==[3,5,9]}")
print(f"   my share of the pool = {MYW:.4f} = 100 throws / {int(1/MYW)*100} = 0.5%: {abs(MYW-0.005)<1e-12}")
print(f"   34% of the field shares my table: {TFRAC[0]==0.34}")
