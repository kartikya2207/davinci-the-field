# THE FIELD — DAY 3 — FINAL ANSWER BLOCK  (v3.2)

    Q1  MIX          Rock 0    Paper 50    Scissors 50            (sums to 100)
    Q2  MARKET       Bid 40    Offer 50                            (spread 10)   <- v3.2: moved from 35/45, see Q2
    Q3  BUCKETS      <30: 0  |  30-34: 0  |  34-38: 0  |  38-41: 25  |  >=41: 75   (sums to 100)
    Q4  MARKETS      M1 BUY   M2 BUY   M3 BUY   M4 BUY
    SKIP             NO on all four

All integers, all constraints checked by script (Q1/Q3 sum to 100, spread = 10).

---

## 1. CENTRAL FORECAST AND PREDICTIVE DISTRIBUTION

**Campus Scissors Day 3: 43.1** (re-weighted ensemble; median 43.4, sd 2.0). 90% band **39.4 – 45.9**.
(v3.1 had 42.4 with MC v2 at weight .5; see v3.2 addendum for why the recruitment family now carries .6.)

    bucket     <30    30-34    34-38    38-41    >=41
    mass       0.0     0.0      2.2     23.4     74.4     (ensemble, 200k draws)

26% of the mass is within ±1 of the 41 boundary; 6% within ±1 of 38. Rock ~32, Paper ~26 (both ±4:
they are the uncertain ones — see Q1).

Component models (each independently hits Day 1 AND Day 2 when fitted, see §5):

    model                                   S3 mean   P(>=41)  P(38-41)   note
    MC v1 (recompute vs naive, no V3)         42.6      81%      18%      3 seeds identical to 0.1
    MC v2 (prompt-variant LLM bloc)           41.5      60%      36%      carries the rotation mechanism
    MC v3 (v2 with 3 vendor sub-blocs)        41.9      70%      29%      less bloc correlation
    trend: linear / saturating / logistic   45.5/43.3/44.4  ~100%    ~0%    no rotation, no level-2
    trend: recruitment (L0->L1 at 30-65%)     43.3      99.9%     0%
    trend: reversal (level-2 defection)       41.8      69%      25%
    ENSEMBLE (v2 .5, v1 .25, trend .25)       42.4      74%      23%

Weights: the MC family retrodicts Day 2 from Day-1-only data (miss −1.0); the trend family cannot be
backtested at all (a trend needs two points). v2 gets the most weight because it is the only model
that contains the mechanism that makes Day 3 different from Day 2 (§2).

Why r-2 understated Day 2 (38.4 vs 41.3): its Poisson-CH ladder forced level-2 mass to rise with
level-1 mass, and level-2 play (66/0/34) drags Scissors down. The two-day fit says level-2 is 2-3%
and static; recruitment is pure L0→L1 at ~49% of the remaining pool per round. Poisson-CH is
rejected outright on Day 2 (L1 error 0.11 vs 0.001 for the free L0/L1/L2 mixture).

## 2. THE STRUCTURAL CHANGE — WHY DAY 3 IS NOT DAY 2 AGAIN

r-2's load-bearing argument was: best-response-to-published-aggregate == best-response-to-uniform for
all six tables, so feedback can only recruit into level 1, never rotate anyone. **True on Day 2. False
on Day 3.** Best response to the published 25.7 / 32.9 / 41.3:

    (3,5,9) 34% mine -> S   margin 83%        (5,3,9) 12% -> S   margin 10%  (was 43%)
    (5,9,3) 24%      -> R   margin 2.0%   <-- WAS PAPER.  Rock 4.073 vs Paper 3.995
    (9,3,5) 14% -> R      (3,9,5) 10% -> P      (9,5,3) 6% -> R

If every level-1 player recomputes against 41.3, the aggregate is R44 / P10 / S46. If they keep
arg-maxing their own payout — which is indistinguishable from recomputing on Days 1-2 — it is
R20 / P34 / S46. **Scissors is 46 either way. Only Rock and Paper move.** That is why Q3 and Q4 are
robust today and Q1 is not.

I also wrote out what a mid-effort LLM tells each table depending on what the student pasted:
  V1 question only            -> argmax payout, bloc R20/P34/S46
  V2 + Day-2 aggregate        -> (5,9,3) "Rock, slight edge / or mix", bloc R44/P10/S46
  V3 + both days, extrapolate -> (5,9,3) Rock by 50%, AND (5,3,9) flips to Rock by 15%, bloc R56/P10/S34
Under V2 or V3 my best response is Paper (5.28 vs 1.93). Under V1 it is Scissors. Whether students
paste the results is **unidentifiable from two days of data** because V1 and V2 gave identical advice
on Day 2. Indirect evidence that they do: on Day 2, ~63% of the field had ≥5 points on >=41, and ~10%
bid above 36 on Q2 — both need the Day-1 number to have been fed in.

## 3. PER-QUESTION

### Q1 — 0 / 50 / 50

Exact crossover (closed form): with S≈42-43, **Paper beats Scissors for my table once campus Paper is
below ~24.5%**, i.e. once ~9.5 points of the field have moved Paper→Rock. The (5,9,3) level-1 mass is
~19% of the field, so the flip needs about half of them to recompute.

Score is linear in weights, so E[score] is linear too, and the two rotation-aware models put the two
pure plays within noise of each other while disagreeing on the sign:

    model         P(Paper best)   E pure S   E pure P   E 50/50   p10 50/50   P(50/50 < 70)
    MC v2            50.2%          88.7       88.5       88.6       78.5         0.3%
    MC v3            56.6%          88.0       92.0       90.0       80.7         0.1%
    MC v1 (no V3)    35.8%          88.6       76.1       82.4       72.2         4.1%

Prior-variant check (MC v2 rerun with the prompt-mix prior pushed to each extreme, 60k each):

    prior on what students paste        S3    P(>=41)  P(38-41)   P(Paper best)   E pure S   E pure P
    V1-heavy (question only, V1≈60%)   42.2     73%      26%          24%           96.0       78.8
    base (V1≈30%)                      41.5     60%      36%          50%           88.7       88.5
    V2/V3-heavy (results pasted, V1≈10%) 40.8   46%      45%          67%           80.7       93.4

The sign of the Q1 answer is decided entirely by this prior. The 50/50 scores 87.4 / 88.6 / 87.1
across the three — flat — while each pure play swings by 15-17 points.

Scenario table at S=42 (x = points of the field moving Paper→Rock):

    shift    0     3     6     9    12    15    20
    pure S  100   100   100   100    85    64    38
    pure P   41    53    69    90   100   100   100
    50/50    70    77    85    95    92    82    69

The 50/50 is **not a hedge that costs expectation** — expectation is flat to ±2 across models that
disagree on the sign — it removes a ~15% chance of a sub-70 leg and a ~5% chance of a sub-50 leg.
Rock is never right (E 56).
**Flip thresholds:** pure Scissors if you believe campus Paper stays ≥ 27 (i.e. < ~35% of the (5,9,3)
level-1 bloc recomputes); pure Paper if you believe it falls ≤ 22 (> ~65% recompute). Neither belief
is supportable from data. If you must go pure, Scissors: it wins in the V1 world where students behave
as on Days 1-2, and its bad case (full rotation, 38-64) needs the *largest* coordinated move.

### Q2 — Bid 35 / Offer 45

Calibration: 36/46 at 41.3 scored 90.4. Under every convention that is "bought every offer ≤36 at a
5.3-point profit, sold nothing". The ~10% who beat me bid above 36. All three execution conventions
reproduce the observation equally (one point cannot separate them), but my field model *over-predicts*
my Day-2 score (96-97 vs 90.4) under all three: the real field is **more aggressive** than a naive
model says. Day 1 (30/40 at 36.9, Q2 ≈ 61-66 by subtraction) says the zero-P&L block sat at ~63,
i.e. 25-35% of the field had negative P&L — aggressive quoters exist on both sides.

The structure that decides Day 3: with everyone quoting a 10-30 spread around 42-44, quotes only cross
in the tails. There are two profit channels and a 10-spread cannot capture both:
  (a) buy from contrarian sellers offering ≤40   — needs a high bid (40)
  (b) sell to bullish extrapolators bidding ≥44  — needs a low offer (44-45)
Channel (b) is the robust one. Sweeping bull share 2-12% × bear share 0-6% × 3 conventions:

    quote    worst   avg    at forecast-3   at forecast   at forecast+3
    34/44     63     69.1        83              70              35
    35/45     63     68.3        68              69              52
    40/50     41     66.4        36              74              92

Bid 40 collapses to 41-43 whenever there are no bears (nothing to buy, and the default bloc — which
*sells* to the bulls at 44 — beats you). 35/45 captures the bull flow at a better price than the
default bloc, ties nobody, and loses only if S ≥ 45 (P ≈ 10%) vs S ≥ 44 for 34/44 (P ≈ 20%).
It is the argmax under my-price, second by 1 point under their-price, and the tightest across ±3.
**Flip thresholds:** forecast ≥ 44 → switch to 40/50 (buy the bears, 92 at +3); forecast ≤ 40 →
34/44. Never bid 42+ (that buys the whole default bloc at 44).

### Q3 — 0 / 0 / 0 / 25 / 75

Two separate Day-2 failures, fixed separately:
(a) Forecast: the machinery now retrodicts Day 2 at 42.3 from Day-1 data (miss −1.0, right bucket).
(b) Allocation: the score is a percentile, the field contains a ~20-30% default bloc (20 on every
bucket → dev 160 whatever happens), a correlated LLM bloc (today: 60-80 on >=41 with 20-30 on 38-41),
sharp players, boundary-splitters, and stale entries. Against that field:
  - 75 on >=41 beats an LLM bloc that put ≤70 there when >=41 hits, and >=41 is unbounded above so
    every overshoot lands in it;
  - 25 on 38-41 beats the default bloc and every entry with <25 there if the result lands 38-41
    (23-36% of the mass under the rotation-aware models) — it turns the Day-2 catastrophe (dev 190,
    37th percentile) into dev 150 and a mid-60s percentile.
The opponent-field model is calibrated: with defaults 30% / LLM bloc 35% / sharp 15% / boundary-
splitters 10% / stale 10% it reproduces my Day-2 37.1 (model 37.3) and my Day-1 ~95+ (model 96.6).
Argmax by run (full 5-tuple grid in steps of 5, E[percentile] against 517 simulated opponents):

    distribution     opp A          opp B          opp C
    MC v1            25/75 79.9     all-in 82.4    20/80 80.6
    MC v2            25/75 71.9     25/75 69.7     25/75 71.1
    ensemble         25/75 77.9     5/95  75.3     25/75 76.3
    trend blend      all-in 96.1    all-in 96.5    —

25/75 is the argmax in 6 of 8 rotation-aware runs and within 1.5 points of the top in the other two.
Under the prior extremes: V1-heavy → 25/75 (77.0, vs 30/70 73.9); V2/V3-heavy (P(38-41)=45%) →
35/65 or 30/70 (67.2, vs 25/75 ≈ 62.5). The two alternatives are symmetric ±3-5 around the base case,
so 25/75 holds at the ensemble belief; **move to 30/70 only if you weight the "results pasted" world
above ~60%.**
All-in on >=41 wins only when P(38-41) < ~8%. The Day-2 shape (0,0,55,40,5) scores 5-20 today.
**Flip thresholds:** all-in >=41 if P(38-41) < 8% (forecast ≥ 44.5); 25/50/25 across 34-38/38-41/>=41
if the forecast drops to ~39.5 (that is the −3 case: 25/75 scores ~40 there, the same failure mode as
Day 2 mirrored). Expected over a 16/68/16 weighting on −3/0/+3, 25/75 still beats 30/70 (74 vs 72).
Boundary mass: 26% within ±1 of 41. The allocation is built for it, not against it.

### Q4 — Buy / Buy / Buy / Buy

All 81 combinations scored by the real clamp-and-ratio function (which reproduces my Day-2 98.0 exactly):

    Buy/Buy/Buy/Buy     99.7   (p5 97.9)
    Buy/Buy/Buy/Leave   96.6   (p5 93.3)
    Buy/Buy/Buy/Sell    86.3

P(S > 40) = 86%. Leave-M4 beats Buy-M4 only if the forecast falls below ~39.5 (at forecast−3 it is
99.2 vs 98.2). M1-M3 are 10+ points clear. As you noted, the clamp-and-ratio structure means M4 is
worth ~3 points either way; not worth a second thought.

### SKIP — none

Reading (i) skip = 0: never. Reading (ii) excluded: skipping Q3 on Day 2 would have turned 325.5/400
into 288.4/300 = 96.1% — but the three days are summed, the percentage is cosmetic, and the 37.1 was
real points. The untouched 20/20/20/20/20 default on Day 2 (dev 160) would have scored ~75 (calibrated
field model, q3cal.py) — nearly 40 points above my 37.1. That is an argument for never leaving a boundary-adjacent bucket empty
(today's 25 on 38-41 is exactly that), not for skipping.

## 4. BACKTESTS

**Forecast.** Fit on Day 1 only (L0 69.4% / L1 27.5% / L2 3.1%; recruitment, inertia, LLM share and
level-2 drawn from the same priors used for Day 3). Day-2 prediction: **S 42.3** (5-95%: 39.0-45.4),
R 26.0, P 31.7. Actual 41.3 / 25.7 / 32.9. Miss −1.0 on S; correct bucket; P(>=41) was 74%.
r-2 predicted 38.4 (miss +2.9, wrong bucket). The trend models cannot be backtested (one point, no
trend); with the true recruitment rate supplied they would have said 41.1.

**Allocator.** Given Day-1 information only, the Day-2 bucket forecast was 74% >=41 / 24% 38-41, and
the percentile optimiser would have entered ~(0,0,0,25,75) → deviation 50 at 41.3, near the top of the
field instead of 37.1. Both halves of the Day-2 loss are reproduced and fixed.

## 5. WHERE THE MODELS DISAGREE

- **S3 level: 41.5 (MC v2) vs 43.9 (trend blend).** The 2.4-point gap is entirely (i) level-2 defection
  of Scissors-holders to Paper (0-6%, mean ~1.5 pts) and (ii) the V3 flip of (5,3,9) off Scissors
  (~1 pt). Nothing in the data settles it: level-2 was 2-3% on Days 1-2 but "level 2" meant Rock then
  and means Paper now. I carry both at 50/25/25; the 25/75 allocation is the argmax under both.
- **Q1 sign: MC v1 says Scissors by 12; MC v2 says tie; MC v3 says Paper by 4.** The flip is the
  prompt-variant prior and the bloc correlation structure — neither identifiable. Settled by refusing
  to pick: the 50/50 has the same expectation under all three and the best worst case.
- **Q2 channel: bid 40 (buy bears) vs offer 44-45 (sell bulls).** Settled by the field's observed
  aggressiveness (10% bid above the crowd on Day 2) and the trend (extrapolators are bullish): the
  bull channel is the fatter one, and bid 40 has a 41-43 floor when bears are absent.
- **Q3: all-in vs 25/75.** Settled by P(38-41): every rotation-aware model puts it at 18-36%, only the
  no-rotation trend models put it below 8%. The rotation-aware models are the ones that backtest.
- **Poisson-CH vs free mixture.** Not a disagreement any more: Poisson-CH is rejected by Day 2.

## 6. TWO BIGGEST UNCERTAINTIES

1. **Do LLM-using students paste the published results into the prompt?** This single behaviour
   decides whether the (5,9,3) bloc rotates to Rock, which decides whether Paper or Scissors is my
   best move (Q1), and it moves S3 by ~1 point (Q3 boundary). V1 (question only) → Scissors 95 vs
   Paper 82. V2/V3 (results pasted) → Paper 92 vs Scissors 82. I cannot tell from two days of data
   because the advice was identical until today. The 50/50 mix is the answer *because* of this.
   What would change it: any direct evidence of how the modal student prompts — e.g. if the results
   are on the same page as the question (→ pasted → lean Paper) or on a separate page (→ lean Scissors).

2. **The size of the bullish tail on Q2 and the ≥44 outcome.** The 35/45 quote profits from players
   who bid ≥45 and loses if S ≥ 45 (P ≈ 10%). If the field has learned to quote wide and nobody bids
   above 44, the offer is dead and I land in the zero block (~50-60th percentile). If S overshoots to
   46+, both my 25/75 and my 35/45 do worse than the aggressive alternatives (all-in >=41, 40/50) —
   though Q3 still scores ~91 in that world.

## 7. SURPRISING THINGS

- **The r-2 "feedback cannot rotate anyone" theorem expires today, by a 2% margin.** Rock 4.073 vs
  Paper 3.995 for the (5,9,3) table at the published Day-2 numbers. A quarter of the field is sitting
  on a coin-flip that an LLM will call one way and a payout-argmaxer the other. The whole Q1 question
  is which of those two the modal student is.
- **Scissors' share is immune to the rotation.** Both the naive and the recompute ladders put S at 46.
  The only things that move S3 down are level-2 Paper defection and the V3 flip of (5,3,9) — together
  ~2 points. That is why the top bucket is robust and the Rock/Paper split is not.
- **Level-2 play is essentially absent and Poisson-CH is dead.** L0/L1/L2 = 69/27.5/3 → 35/63/2.
  One round of feedback recruited half of the remaining level-0 pool. Camerer's e^-1.5 = 22% level-0
  does not describe this field; it went from 69% to 35% in a day.
- **On Q2 the untouched default is a money-maker today.** Bullish extrapolators bidding ≥44 cross the
  default's offer at 44, so the 25-30% default bloc profits and lands ~70th percentile. 35/45 rides
  the same flow at a better price and without tying 145 people.
- **A 25-point insurance leg on Q3 costs ~1 percentile point in the good state and buys ~30 in the bad
  one.** Because the score is a rank against a field with a default bloc at 20-on-every-bucket, the
  marginal value of the 21st-25th point on the runner-up bucket is enormous and the marginal cost of
  the 76th-100th point on the modal bucket is nearly zero.
- **Q1 was a 100 twice, and today it is a coin flip.** The (3,5,9) table was the most Scissors-robust
  of the six for two days; today it is the table with the most to lose from a rotation it cannot see
  coming in its own numbers (S margin 83% at the published aggregate).

## FILES
core.py (formula, verified vs worked example + both days) · br_analysis.py · fit2day.py · crossover.py ·
mc3.py / mc4.py / mc5.py (agent MC v1/v2/v3) · trend.py · backtest.py · ensemble.py · q1sens.py ·
llm_chain.py · q3cal.py (opponent-field calibration vs Day-2 37.1) · q2cal.py / q2sim.py / q2v2.py · q3rank.py · q4.py · logs q3_*.log, mc*_s*.log

## LOG
- 2:40 v3.1: prior-variant MCs (V1-heavy / V2-V3-heavy), Q2 second seed (ordering unchanged), Q3 on both variants. No change to the block.
- 0:25 v1. 1:15 v2 (Q1→50/50, Q3 25/75, Q2 40/50). 2:10 v3: Q2→35/45 after bull/bear sweep; ±3 checks;
  sub-bloc MC; ensemble Q3; backtests written up.


## v3.2 ADDENDUM — five jobs and one new finding (deltas only)

**Job 4 (page structure, from the PDFs).** The Day-3 play page prints "What IIT Bombay threw on Day 2:
R 25.7% P 32.9% S 41.3%" in a highlighted card directly ABOVE Question 01, and the landing page says
"The campus landed at 41.3% scissors" in display type. Anyone who pastes or screenshots the page hands
the LLM the numbers. Also confirmed from the pages: Q1 default = even split (34/33/33), Q2 default 34/44,
Q3 default 20 each, Q4 default Leave×4 (so the Q4 field is mostly zeros — irrelevant to my absolute score).
This pushes the prompt prior toward "results pasted" → P(Paper best) 67% (mc4 V2/V3-heavy, mc6).

**New finding that pushes back.** Every naive payout-argmax move scored ≥98 on Day 2:
(3,5,9)S 100 · (5,9,3)P 98.1 · (9,3,5)R 100 · (5,3,9)S 100 · (3,9,5)P 100 · (9,5,3)R 100.
The (5,9,3) bloc is being asked to abandon a 98 for a 2% computed edge. Humans do not; an LLM
might. Rotation now comes only from LLM users who override a 98 — r_eff ≈ 0.35-0.5, P(Paper best) ≈ 35-50%.
Net of the two pieces of evidence: still a coin flip. Q1 stays 0/50/50. Tilt table (E / p10 / P<70):

    model               0/100      50/50      60/40      70/30      100/0    P(P best)
    V1-heavy            95.9/83    87.4/77    85.7/72    83.9/67    78.8/54     24%
    base                88.7/65    88.6/79    88.6/78    88.6/74    88.5/63     50%
    V2/V3-heavy         80.8/53    87.1/76    88.3/79    89.6/81    93.4/73     67%
    near-tie=split      84.1/60    89.3/80    90.4/82    91.4/84    94.6/78     67%
    no-rotation (v1)    88.7/56    82.3/72    81.1/68    79.8/62    76.0/46     36%
0/60/40 is within 1.5 of 0/50/50 in every world — a free tilt toward the evidence if wanted. Not >3: no change.

**Job 2 (ensemble weights).** The recruitment arithmetic (L0 35%→17.9%, S=43.5) is right and it is the
best-supported point estimate; the two mechanisms that pulled MC v2 below it — the V3 flip of (5,3,9) off
Scissors and a level-2 Paper defection with prior mean 4% vs fitted 2-3% — deserve less than half the weight.
Re-weighted (trend .6 / v2 .25 / v1 .15): **S3 = 43.1**, P(≥41) 85%, P(38-41) 13%, P(≥44) 35%, P(≥45) 15%.
Original: 42.4 / 74% / 23%. Adopted 43.1.

**Job 5 (ceiling).** Hard. Scissors-holders are 46% of the field and no non-S table best-responds to
Scissors under any anchor (uniform, Day 1, Day 2, extrapolated). Non-level-1 mass plays S at 33% (defaults)
or 41% (inertia) — both below 46. S3 > 46 needs a bloc outside the S-holders to pick Scissors on momentum.
L1=90% → 44.7; L1=100% → 46.0. MC 97.5th pct 45.2-45.9. Treat 46 as the cap; the linear-trend 47 is the
one model with no ceiling and is wrong above 46. Consequence: P(S≥45) ≤ ~15-25% — bounds the offer-45 risk
and the upside of all-in ≥41.

**Job 3 (Q3 on the re-weighted ensemble).** all-in 84.0 · 20/80 82.4 · 25/75 80.7 · 30/70 75.2.
20/80 vs 25/75 = +1.7 here, +5.3 on pure trend, but −6.1 (base), −8.5 (V2/V3-heavy), −11 (forecast−3).
Under 2 points on the preferred distribution, and 25/75 is the better minimax. **No change: 25/75.**

**Job 1 (Q2 grid, bids 34-42 × spread 10-14, 16 bull/bear scenarios × 3 conventions).**
The 36-39 range is not where the answer lives — **36-37 is the trough** (36-39th percentile: dead on both
sides, and the zero-P&L block sits low because the tails make money). Full table:

    centre 42.4:  34→68  35→68  36→61  37→36  38→59  39→66  40→68  41→65  42→59
    centre 43.5:  34→56  35→63  36→61  37→39  38→62  39→70  40→74  41→73  42→70
Spread is irrelevant for bid ≥38 (offer dead); at bid 34-35 only spread 10-11 works (offer 44-45).
By scenario: with zero bears (nobody offers ≤40) 34/45-35/45 wins 71-76 and 40/50 scores 43-47; with
bears ≥2% of the field, 40/50 wins 72-91 and 35/45 scores 57-70. At centre 43.5, 40/50 wins 12 of 16
scenarios and averages +11 over 35/45; at 42.4 they tie. Bears ≥2% (Day-1-anchored quoters, reversal
believers offering 38-40) are more likely than not; Day 1 showed 25-35% of the field with negative P&L,
so the field does cross. **Changed: 35/45 → 40/50.** Loss case: S < 40 (P ≈ 8%). 39/49 is the cautious
variant (−2 to −4 expected, loses only below 39).
