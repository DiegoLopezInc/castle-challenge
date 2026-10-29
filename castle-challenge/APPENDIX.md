# Appendix — Castle Challenge Solver

**Chosen plays**

| game | allocation | rationale |
|---|---|---|
| Game 1 (head-to-head, avg castle points) | `[1, 0, 14, 39, 46]` | one-up the sophisticated consensus; robust across field models |
| Game 2 (4-way, rank payouts 9/4/2/0) | `[0, 33, 0, 0, 67]` | best worst-case-across-fields; locks castle 2 + overbids castle 5 |

## Methodology

### Game 1 — `castle_solver.py`, `scenarios.py`

- **Score decomposition.** For a fixed opponent population, expected castle
  points factor by castle: `E = Σ_i v_i·(Pr[b_i < a_i] + ½·Pr[b_i = a_i])`,
  using each castle's marginal distribution. The best response to any
  population is therefore a knapsack DP over (5 castles) × (0–100 soldiers) —
  no enumeration of the ~4.6M legal allocations needed.
- **Exact worst case.** For a fixed allocation, the opponent minimizing your
  score is found by a min-convolution DP over the same grid (exact over all
  allocations, ~2 ms). Used for the maximin analysis and local hill-climbing
  (`polish`).
- **Field models.** Plausible opponent pools: uniform compositions, noisy
  heuristics (uniform split, proportional-to-value, cheap-castle grab,
  top-stack, exploit), Dirichlet mixtures, and a self-play field produced by
  300 rounds of fictitious play (add best-response each round).
- **Decision criteria.** Expected score under an equal-weight mixture of
  archetypes (Bayes), max regret across archetypes (minimax regret), and
  best response to the converged self-play field (equilibrium response).

### Game 2 — `castle_game2.py`

- **Monte Carlo payout evaluator.** For each simulated group, all 4 players'
  castle-point totals are computed including k-way splits (points scaled ×12
  so splits among ≤4 players are integer-exact). Your payout depends only on
  how many opponents scored strictly above / equal to you, via a precomputed
  4×4 tie-split table on payouts [9,4,2,0]. Vectorized; 60k–200k groups per
  evaluation.
- **Evolutionary search.** Population of ~160–192 allocations, fitness =
  expected payout vs the field prior, single-soldier-transfer mutations,
  elitist selection, common random numbers across candidates for stable
  comparisons. Converges in ~30–60 generations.
- **Field prior.** Mostly non-naive: noisy `[1,3,13,38,45]` and
  `[1,0,14,39,46]` clouds + self-play population + ~30% naive tail. Robustness
  checks add counter-strategy injections (`[17,17,17,49,0]`, `[16,16,16,0,52]`,
  `[0,0,0,45,55]`/`[0,0,0,47,53]` one-uppers).

## Results

### Game 1 — `[1,0,14,39,46]`, expected castle points (of 15; fair share 7.5)

| field | score |
|---|---|
| non-naive prior | 9.14 |
| naive-only | 9.21 |
| self-play converged | 9.53 |
| balanced | 9.22 |
| proportional-to-value | 9.29 |
| top-heavy | 6.88 |
| cheap-grab | 9.87 |
| uniform random | 9.52 |
| monoculture `[1,3,13,38,45]` | **12.50** (one-ups the consensus) |
| worst case vs any opponent | 3.00 |
| vs exploit `[2,0,0,49,49]` | 4.00 |

### Game 2 — `[0,33,0,0,67]`, expected payout (of 9; fair share 3.75)

| field | E[payout] | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|---|
| non-naive prior | 7.74 | .78 | .18 | .02 | .03 |
| naive-only | 7.19 | .68 | .25 | .03 | .04 |
| self-play converged | 7.24 | .67 | .30 | .01 | .02 |
| prior + 29% counters | 7.87 | .80 | .17 | .01 | .02 |
| 100% counters | 7.81 | .76 | .24 | .00 | .00 |
| monoculture `[0,33,0,0,67]` | 3.75 | .25 | .25 | .25 | .25 |

Worst-case across all tested fields: **7.13** (vs 3.13 for the
higher-prior-mean alternative `[0,0,0,46,54]`).

### Key structural results

- Game 1 pure-strategy maximin is ~4/15: an equal-budget opponent can always
  buy your three cheapest castles and overbid your most expensive weak
  castle. Balanced splits are the *worst* plays (worst case 1.0).
- Game 2 `[0,0,0,46,54]` cannot lose both top castles (47+55=102>100), but the
  flank shape `[0,33,0,0,67]` dominates under counter-heavy fields: it wins
  castle 2 above the ~17–21 counters invest there and overbids castle 5 at
  67. Counter strategies are self-defeating — identical counters tie each
  other on contested castles and hand the flank play first place.
- Game 2 scoring is rank-based, not points-based: conceding 3 castles costs
  nothing if the 2 castles you win total more than any opponent's haul.

## Files

- `castle_solver.py` — Game 1: best-response DP, exact worst-case DP,
  double-oracle maximin search. `python3 castle_solver.py [a b c d e]`
- `scenarios.py` — Game 1: field archetypes, regret matrix, Bayes /
  minimax-regret / self-play recommendations
- `castle_game2.py` — Game 2: Monte Carlo payout evaluator + evolutionary
  search. `python3 castle_game2.py`

## Caveats

Expected values are conditioned on the field models above; the true applicant
pool is unobservable. Both picks were chosen to minimize regret across
plausible field compositions rather than maximize score under a single guess.
Monte Carlo standard error at 80k groups is ≈ ±0.02 payout points.
