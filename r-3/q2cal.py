"""Q2 calibration: I quoted 36/46 on Day 2, actual 41.3, scored 90.4 percentile.
Under each convention, who beats me?  Also Day 1: quoted 30/40, actual 36.9, Q2 approx 61-66."""
import numpy as np
print("=== analytic ===")
a=41.3; myB,myO=36,46
print("Day2: I buy from every offer<=36. Profit/unit: mine=%.1f; theirs>=%.1f (their offer<=36). Sell side: needs bid>=46: none."%(a-myB,a-36))
print("Under THEIRS: anyone with bid in (36, ~41] strictly beats me (superset of my fills, each profitable).")
print("   => 90.4 percentile means ~9.6%% of field bid above 36 (ties at 36 split).")
print("Under MINE: bid b beats me iff (41.3-b)*n(offers<=b) > 5.3*n(offers<=36). Needs offer density above 36.")
print("Zero-trade players (most of field) score 0 -> below me under all conventions. Negative: anyone who SOLD at offer<41.3 to an aggressive bid.")
print("\nDay1: quoted 30/40 at actual 36.9 -> bid 30 fills ~nothing -> P&L ~0 -> mid-tie of the zero block.")
print("   Day1 total 360.9 = Q1 100 + Q4 100 + Q2 + Q3(all-in true bucket).  Q2+Q3=160.9.")
print("   If Q3~95-100 then Q2~61-66: zero-P&L block sits at ~61-66 percentile -> ~25-35%% of field had NEGATIVE P&L on Day 1.")
print("   Negative P&L on day 1 (actual 36.9) = sold at offer<=36.9 into aggressive bids, or bought at >=37 -> aggressive quoters exist on both sides.")
