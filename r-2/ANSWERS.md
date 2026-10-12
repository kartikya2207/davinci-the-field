# THE FIELD — DAY 2 — FINAL ANSWER BLOCK

    Q1  MIX          Rock 0    Paper 0    Scissors 100          (sums to 100)
    Q2  MARKET       Bid 37    Offer 47                          (spread 10)
    Q3  BUCKETS      <30: 0  |  30-34: 0  |  34-38: 55  |  38-41: 40  |  >=41: 5   (sums to 100)
    Q4  MARKETS      M1 BUY   M2 BUY   M3 BUY   M4 LEAVE
    SKIP             NO on all four.

Central forecast: campus Scissors 38.4% (median 38.3, 90% band 35.1-41.5).
Day 1 was 36.9. My model says Scissors goes UP, not flat.

---

## 1. Q1 — 0 / 0 / 100 Scissors

Score is linear in my weights (`sum_m w_m v_m / max_m v_m`), so the optimum is always a
pure play on the move with the highest `E[v_m / v_max]`. Across 40,000 simulated Day-2
fields: **E[score] Scissors 99.60, Paper 73.67, Rock 48.74**. Scissors is the argmax in
94.9% of draws, and even in the 5.1% where it is not, it still scores 61-90, so the
downside of the pure play is tiny — 5th percentile 99.71, worst case over 40k draws 60.8.
Every hedge strictly loses: 0/10/90 -> 97.0, 0/20/80 -> 94.4, 5/5/90 -> 95.8.

**Flip thresholds (my table 3/5/9, computed exactly):**
- Scissors -> **Rock** requires `s_S >= 0.497` (at s_P=0.24) rising to `0.519` (at s_P=0.34).
  Call it **campus Scissors above 50%**. My 95th percentile is 41.5. Not reachable.
- Scissors -> **Paper** requires a Rock cascade: `s_R >= 0.342` *and* `s_P <= 0.25`.
  At s_P >= 0.30, Paper never beats Scissors anywhere on the simplex.
  My 95th percentile for s_R is 36.6, and P(s_R > 46%) = 0.0000.

Table 3/5/9 is structurally the **most Scissors-robust of the six tables** — see the
surprises section.

## 2. Q2 — Bid 37 / Offer 47

The binding structure nobody stated: with 467 players all quoting a 10-30 spread around
~37, **quotes almost never cross**. Bids cluster 30-35, offers 40-48. To trade at all you
must be a deliberate one-sided aggressor. That makes this a bid-side problem — the offer
side is unreachable without crossing the 34/44 default bloc, and selling at 34 into a 38.4
settlement is a 4-6 point loss multiplied by ~140 counterparties.

So: lift the low tail of offers, and price the bid just under the settlement.
**Bid 37 is the argmax under all three execution conventions simultaneously** —
rank-score 66.7 (my price) / 71.7 (their price) / 69.2 (midpoint). It is also the maximin
choice. The convention changes the *magnitude* of the edge, never the location of the
optimum: "their price" pays me the counterparty's low offer, so it is the most generous;
"my price" is the harshest and sets the ceiling on how high I can bid. Since all three
agree, the unresolved mechanic does not actually need resolving.

The cliff is on the upside and it is brutal — one tick too high and you buy the default bloc:

    bid 35 -> 54.5 | 36 -> 61.1 | 37 -> 66.7 | 38 -> 56.8 | 39 -> 35.7 | 40 -> 20.5 | 42 -> 2.9

Offer 47 is the tightest legal offer given bid 37 (spread exactly 10). It sits safely above
my 95th percentile of 41.5, so **any fill on it is pure profit** — a free lottery ticket on
an aggressive buyer, with no scenario in which it fills at a loss. That is why the spread is
10 and not wider: widening only pushes the offer further out of reach, and the page's warning
about tight spreads is about crossing risk, which here lives entirely on the bid.

Default 34/44 scores 51.6 (it trades nothing and lands mid-tie). The research pass's
33/45 scores **50.2 — worse than leaving the page untouched.**

**Flip threshold: bid 37 is right while E[Scissors] >= ~37.5. At 37.0 bid 36. At 36.0 bid 35.
Never bid 39+ under any belief in this range.**

## 3. Q3 — 0 / 0 / 55 / 40 / 5

This is the question where the correct answer is *not* the deviation-minimising one, and
the research pass and the naive arithmetic are both wrong.

Raw expected deviation is minimised by all-in on the modal bucket — that much is algebra.
But the score is a **percentile against 467 opponents**, and a ~42% LLM bloc piles onto the
same modal bucket. Against a bloc:
- If the bloc is right, they all tie at deviation 0 and I cannot beat them whatever I do —
  going all-in on their bucket merely *joins* the tie, mid-rank ~50.
- If the bloc is wrong, they all tie at deviation 200, and I beat **all 42% of them at once**
  with any weight at all on the true bucket.

That asymmetry makes hedging strictly better on rank even though it is worse on deviation.
Simulated, 467 opponents, 2,500 field draws, bloc modelled both as pure-modal and as the
research pass's own 0/18/82/0/0:

    (0, 0, 55, 40,  5)  E[percentile] = 60.03   <- chosen
    (0, 0, 60, 40,  0)  E[percentile] = 59.43
    (0, 0, 60, 35,  5)  E[percentile] = 59.15
    (0, 0,100,  0,  0)  E[percentile] = 49.88   (pure modal — joins the tie)
    (0, 0,  0,100,  0)  E[percentile] = 37.61
    research (0,18,82,0,0) — not on the efficient frontier at all

**+10 percentile points over the pure play.** The same allocation is the argmax under my own
bucket probabilities *and* under the blended (my-model / research-model) probabilities, so it
is not tuned to one forecast.

Blended bucket masses used: 2.3 / 8.9 / 46.1 / 34.3 / 8.5 (my MC weighted 55%, the research
pass's distribution 45%). Note the 18 points the research pass put on 30-34: my model gives
that bucket 1.4% standalone and 8.9% blended. It is close to dead weight; those points belong
on 38-41, where my central forecast actually sits.

**Flip threshold: the 55 pays if campus Scissors lands 34-38, the 40 pays at 38-41. The
allocation should flip to 40/55/5 the moment you believe E[Scissors] > 38.5.**

## 4. Q4 — Buy M1, Buy M2, Buy M3, LEAVE M4

Simulated all 81 combinations under the full predictive distribution, scored by the **actual
score function** (clamped profit / total positive profit available), not by profit:

    (Buy,Buy,Buy,Leave)  E[score] = 99.58   p5 = 97.3   <- chosen
    (Buy,Buy,Buy,Buy)    E[score] = 96.17   p5 = 89.2
    (Buy,Buy,Buy,Sell)   E[score] = 93.83   p5 = 87.8   <- research pass
    (Buy,Buy,Leave,Leave)E[score] = 84.68

**There is no convexity rescue.** The clamped-ratio scoring does not make the negative-EV M4
position worth taking; Leave beats Sell by 5.7 expected score points. The mechanism is that
when I Leave M4 and the settlement lands inside 35-40, M4 contributes **zero to both the
numerator and the denominator**, so I score a clean 100. Leave is not a neutral abstention
here — it is the scoring-optimal action inside a spread. Taking either side can only add a
negative numerator against an unchanged denominator.

Robust across every stress: 99.6 (my mean 38.4) / 99.5 (research mean 37.0) / 99.0
(pessimistic 36.0) / 98.6 (cascade 40.4). It is top-ranked in three of four and within
0.2 of top in the fourth.

**Flip thresholds:** Buy M4 becomes right only if campus Scissors clears 40 —
P(actual > 40) = 21.2%, not enough. Sell M4 becomes right only below 35 — P = 4.6%.
M1/M2/M3 buys flip to sells at 10 / 20 / 30 respectively; all three are 8+ points clear,
so nothing there is close.

## 5. SKIPPING — do not skip anything

Under reading (i), skip = 0: obviously never, every answer above is worth 60-100.

Under reading (ii), skip excluded from numerator and denominator: skipping question Q is
right only if `E[score_Q] < E[mean of the questions I do answer]`. My four expected scores
are ~99.6 / ~67 / ~60 / ~99.6, mean ~81.6. Q2 (67) and Q3 (60) sit below that mean, so under
reading (ii) skipping both would mechanically *raise my Day-2 percentage*.

**It would still be wrong, and this is the load-bearing point: the three days are SUMMED, not
averaged.** I am 30.0 points behind on a 1200-point total. Skipping converts a near-certain
+60 of summed score into +0 in exchange for a cosmetic ratio that is not what is being added
up. At rank #16 chasing a 30-point deficit I need raw points, and the argument for skipping
survives only under a scoring rule the game does not use. Answer all four.

---

# WHERE I DISAGREE WITH THE RESEARCH PASS

**A. Its Day-1 fit is wrong, and the error is large.** It fitted a two-type model (naive vs
sharp) and concluded only 25-33% of the field played sharp. That model **cannot fit Day 1** —
the best two-type fit is R 30.7 / P 33.5 / S 35.8 against the observed 30.7 / 32.5 / 36.9, an
L1 error of 0.0203, missing Paper by a full point. Allowing a real cognitive hierarchy fits
Day 1 **almost exactly**: L0 22.0% / L1 49.0% / L2 20.5% / L3 8.5% -> 30.66 / 32.49 / 36.84,
error 0.0010. That is **78% of the field non-naive, not 28%.** (The fitted L0 share of 22.0%
is an independent hit on Camerer/Ho/Chong's e^-1.5 = 22%.) A one-parameter Poisson-CH fit
gives tau = 0.395, error 0.0124 — worse than the free hierarchy, far better than two-type.
**Consequence: the field is sharper than the research pass believes, so Day 2 climbs further.**

**B. The "(a) and (b) cancel" claim fails, because level 1 and level 2 are not symmetric about
Day 1.** I computed the ladder explicitly. Level-1 aggregate is 20/34/46. Level-2 is 66/0/34.
Level-3 is 0/100/0. Level-4 is 0/0/100. **It is a period-3 cycle with no interior convergence**
— exactly the Cason/Friedman/Hopkins instability, which is the one piece of the research
pass's own literature review it then ignored when predicting a flat repeat. Climbing the
ladder does not return you to Day 1; it rotates you around it. With most of the Day-2 climb
running L0 -> L1 (that is where the mass is) rather than L1 -> L2, the net is **Scissors up**.
**My Day-2 forecast is 38.4, not 37.** That single number is our main disagreement and it is
what moves Q3's weight onto 38-41 and lifts the Q2 bid to 37.

**C. Its stated crossover claims are wrong, in a way that flatters its own answer.** It says
"the Scissors->Paper crossover needs s_S well above 0.55." The Scissors->Paper crossover has
essentially nothing to do with s_S — it is driven by **s_R**, and it fires at s_R = 0.342 when
s_P <= 0.25. It reached the right recommendation (pure Scissors) by watching the wrong
variable, which means its stress tests cannot detect the scenario that would actually break it.
The number to monitor is Rock, not Scissors.

**D. Q2 — your error confirmed, and the recommendation is worse than doing nothing.** My
forecast of 38.4 (and even its own 37) is *below* the default mid of 39, so a quote centred
below the default is right in direction — but it then set bid 33, too low to trade at all, and
offer 45, unreachably high. 33/45 scores 50.2 against the untouched default's 51.6.

**E. Q4 — your error confirmed, and the convexity defence does not exist.** Leave beats Sell
by 5.7 expected score points, in every stress case, scored on the real objective.

**F. Q3 — your error confirmed, but the fix is the opposite of what the arithmetic implies.**
Its 18 points on 30-34 do lower the expected deviation-score, as you said. But pure
concentration is *also* wrong once the score is a rank against a correlated bloc. The right
answer is a hedge — just a different one, shifted a bucket to the right.

**G. Your SD-0.35 concern is correct and it matters.** If most players submit pure strategies,
the per-player weight is Bernoulli-like. My agent-based model generates the sampling noise
endogenously from 467 discrete agents and gives **SD = 1.98 points** on the Scissors share,
between the research pass's 1.6 and the 2.2 implied by SD 0.48 — but with genuinely fatter
tails from the correlated-bloc structure, which no binomial SE can capture.

# TWO BIGGEST UNCERTAINTIES

**1. Whether the field climbs to level 1 or overshoots to level 2.** This is the whole game.
Level-1 mass pushes the aggregate toward 20/34/46 (Scissors up); level-2 mass pushes it toward
66/0/34 (Rock up, Paper annihilated). My model has the L0 -> L1 flow dominating, giving 38.4.
If instead the LLM bloc reasons one step further — and frontier models are plainly capable of
the level-2 step here, which is exactly what the "LLMs sit at level 0-1" literature would
under-predict — the aggregate rotates toward Rock. **This does not break Q1** (my table keeps
playing Scissors until s_R > 0.34) **but it does break Q3**: weight would need to move back to
34-38 and 30-34. Watch Rock, not Scissors. If you have reason to think the field is very sharp,
shift Q3 to (0, 5, 55, 35, 5).

**2. What the other 466 actually quote on Q2.** The optimum bid is 37 under all three
execution conventions, so the unresolved mechanic turns out not to matter for the decision —
that is the useful finding. What does matter is my model of the opponent quote distribution.
If the field is far more aggressive than I assume, with a real mass bidding 40+, then bid 37
stops being the top of the crossing tail and the offer side comes alive, favouring something
wider like 34/48. I cannot resolve this before the lock.

# SURPRISING THINGS

- **My payoff table is the best of the six for this exact field, and by a wide margin.**
  Table 3/5/9 is the *only* one of the six whose best response is still Scissors at level 2.
  At the level-2 aggregate (66/0/34) every other table best-responds to Rock; mine gives
  Rock 3.07 against Scissors 4.31 and stays put. The 34% of the field holding my table are my
  direct competition for the same points, but the table itself is a gift.

- **Day-1 feedback is informationally worthless for Q1, and it is provable.** I computed the
  best response of all six tables to the published 30.7/32.5/36.9 and to uniform thirds.
  **They are identical for every single table.** The value multipliers come out at 0.663 /
  0.534 / 0.525 — close enough that argmax(payout) survives unchanged. So the one round of
  public feedback that the entire Day-2 premise rests on **cannot change any level-1 player's
  move.** It can only recruit more players into level 1. That is the mechanical reason Scissors
  rises rather than staying flat, and it is the core of my disagreement with the flat-repeat
  forecast.

- **Leave is not passive on Q4.** Because the denominator is *available* positive profit,
  leaving a market that has no positive profit available deletes it from the denominator
  entirely and lets the remaining three score a clean 100. Leave inside a spread is the
  aggressive play, not the timid one.

- **The Q2 default is a trap with ~140 people standing in it.** Roughly 30% of the field will
  leave 34/44 untouched. It trades nothing, so they all tie at zero P&L and land mid-pack —
  harmless. But anyone who bids 44 or offers 34 to "get filled" transacts against all 140 of
  them at once at a 4-6 point loss per unit. The single most destructive thing available on Q2
  is being one tick too aggressive.

- **Concentration on Q3 does not win when everyone else concentrates too.** Pure modal scores
  49.9 percentile — dead average — because being right ties you with the bloc and being wrong
  ties you with them as well. All of the rank value lives in the hedge.
