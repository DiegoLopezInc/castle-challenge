# poker-bot-comp

Local training ground for **No Limit: A Bot Building Tournament**
(Sequoia × OpenAI × Citadel). See `PREP.md` for dates/deadlines/checklist
and `comp-rules.rtf` for the official rules.

## What's here

```
framework/
  cards.py      deck + pure-Python 7-card evaluator (~microseconds)
  engine.py     8-max NLH engine: blinds, side pots, all-ins, min-raise
                rules, 200-hand cap, per-decision timing (1 ms budget)
  icm.py        Malmuth-Harville ICM (subset DP) for hand-cap scoring
  agents.py     Agent protocol + opponent zoo
  simulate.py   SNG runner + CLI with finish dist + timing stats
example_bot.py  safe-baseline template (conservative push/fold)
```

Everything is **stdlib-only** — the competition sandbox has no guaranteed
packages, no network, a 100 MB file cap, and a 1 ms decision budget, so
practice under the same constraints.

## Rules fidelity

| rule | implemented |
|---|---|
| 8-bot tables, random seats | `run_table` shuffles seats |
| 2000 chips, blinds 10/20 up every 25 hands | `Table(..., blind_schedule=fn)` — configurable, real schedule is day-of |
| 200-hand cap → ICM split | `table.run` + `icm_equities` over remaining point slots |
| finish-position points | `--points` CLI arg (day-of structure) |
| 1 ms decision budget | measured per decision; `--enforce-timing` folds/checks on breach |
| error → fold-or-check | illegal/error actions → check if free else fold |
| state resets per table | fresh agent instance per table (or `.reset()`) |
| T1–T6 weighting 1×–6× | `WEIGHTS` in simulate.py |
| filler bots for non-÷8 fields | n/a locally — zoo fills seats |

## Usage

```bash
# 200 tables, 8 cores, your bot vs the zoo
uv run python -m framework.simulate --bot example_bot.py --iters 1600 --jobs 8

# strict 1 ms enforcement
uv run python -m framework.simulate --bot example_bot.py --iters 200 --enforce-timing

# tests
python3 framework/tests/test_framework.py
```

A bot is a file defining `make_agent()` (or a class `Agent`) returning an
object with `act(obs) -> (kind, amount)`. `obs` keys are documented in
`agents.py` / `engine.py._obs`. `amount` is a raise-to total for `raise`.

## About "SOTA bots" and Kaggle

Honest assessment: real SOTA poker AI (Pluribus, Libratus, DeepStack) is
**not open-source**, and Slumbot's API needs network (banned in the sandbox).
Kaggle is mostly **hand-history datasets** — useful for modeling humans,
but your opponents here are exclusively other bots (rule 3.9), so a diverse
bot zoo is the right sparring partner. The zoo (`make_zoo`) mixes:

- `CallingStation` / `RandomBot` — weak field floor
- `HeuristicBot` (TAG) / `TrappyBot` (LAG-ish) — median student bot
- `PushFoldBot` — ICM-flavored shove/fold; near-optimal SNG endgame and
  likely the toughest archetype you'll face

If you want stronger sparring later, `rlcard` (pip) ships NLH environments
and NFSP-style agents that can be wrapped in the `Agent` protocol — but
expect them to be beatable by a good push/fold + ICM bot anyway.

## Day-of advice baked into the design

- The variant is revealed at build time — keep strategy logic in swap-in
  modules (push/fold charts, ICM endgame, fast evaluator) that port to any
  variant in minutes
- 1 ms is *tight*: profile `decision time` output; precompute anything
  expensive in `__init__` (constructor runs once per table, decisions don't)
- The safest baseline always returns a legal check/call — never bust to an
  exception (errors fold you)
