# No Limit: Bot Building Tournament — Prep Tracker

**Status: APPLIED** — awaiting regional assignment / acceptance.
Register: https://pokerbot.sequoiacap.com

## Key dates (2026)

| event | date | registration deadline |
|---|---|---|
| Menlo Park, CA | Fri Nov 6 | Tue Nov 3, 11:59 PM PT |
| Cambridge, MA | Mon Nov 9 | Fri Nov 6, 11:59 PM ET |
| Pittsburgh, PA | Thu Nov 12 | Mon Nov 9, 11:59 PM ET |
| Finals, SF (invite only) | Sun Nov 15 | — |

Check-in 5:30 PM, event 6:00–9:00 PM local. In person, full event.

## Hard constraints (from comp-rules.rtf)

- Poker variant revealed **day-of** — framework must be variant-agnostic
- 60-min build phase; Python bot implementing provided `Agent` interface
- **1 ms per decision**; error/timeout → auto-fold (or check if available)
- 100 MB max file size; no network / external APIs / OpenAI calls in bot
- Bot state resets each table; opponents anonymized
- One submission every 5 min; latest accepted = active bot
- 8-bot SNGs × 10,000 tables per scored tournament; 200-hand cap
- 2,000 starting chips; blinds 10/20, increase every 25 hands (schedule day-of)
- Points by finish position (structure day-of); ICM allocates points at hand cap
- Filler bots (cloned submissions) fill non-multiple-of-8 fields, earn nothing
- Final score = Σ T_i × i (later tournaments weighted more: 1–6×)
- Can keep editing/submitting between scored tournaments

## Prep checklist

- [ ] Confirm which regional, block calendar + travel
- [ ] Student/eligibility docs ready (enrolled or class of 2025/2026, 18+)
- [ ] Framework: run `framework/simulate.py` against opponent zoo
- [ ] Practice drill: 60-min timed build with a surprise variant
- [ ] Day-of kit: laptop, Python env, this repo, pre-written strategy modules
- [ ] Reusable strategy pieces to have ready:
  - preflop push/fold charts (by stack depth in BB)
  - ICM-aware endgame logic (huge edge at SNG bubbles)
  - fast hand evaluator (1 ms budget!)
  - fallback: always-legal check/call to never bust on errors
