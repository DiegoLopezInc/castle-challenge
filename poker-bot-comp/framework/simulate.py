"""Competition-format simulator: 8-bot SNGs, scored like rules section 4.

- each iteration: your bot + 7 opponents, random seats, play to one winner
  or the 200-hand cap
- eliminated bots score their finishing position's points; survivors at the
  cap split the remaining point slots by ICM
- T1..T6 weighting (x1..x6) applied when you run scored-tournament series

CLI:
    uv run python -m framework.simulate --bot example_bot.py \
        --iters 200 --points 9,6,4,2,0,0,0,0 --jobs 8
"""

import argparse
import importlib.util
import random
import sys
import time
from multiprocessing import Pool

from .agents import make_zoo
from .engine import Table
from .icm import icm_equities

DEFAULT_POINTS = [9, 7, 5, 4, 3, 2, 1, 0]  # placeholder: real payouts day-of
WEIGHTS = [1, 2, 3, 4, 5, 6]


def load_agent_factory(path):
    """Import a bot file; return a zero-arg factory producing a fresh agent.

    The file must define make_agent() -> agent, or a class named Agent.
    Fresh instance per table = state reset (rule 3.8)."""
    spec = importlib.util.spec_from_file_location("user_bot", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_bot"] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_agent"):
        return mod.make_agent
    if hasattr(mod, "Agent"):
        return mod.Agent
    raise ValueError(f"{path}: define make_agent() or class Agent")


def run_table(agents, points, rng, max_hands=200, enforce_timing=False):
    """One 8-bot SNG with fresh agent instances. Returns (points, table)."""
    agents = list(agents)
    rng.shuffle(agents)
    for a in agents:
        if hasattr(a, "reset"):
            a.reset()
    table = Table(agents, rng=rng, enforce_timing=enforce_timing)
    _, survivors = table.run(max_hands=max_hands)
    pts = [0.0] * len(agents)
    if not survivors:
        for p in table.players:
            pts[p.seat] = points[p.finish - 1]
    else:
        k = len(survivors)
        stacks = [p.stack for p in survivors]
        ev = icm_equities(stacks, points[:k])
        for p, e in zip(survivors, ev):
            pts[p.seat] = e
        for p in table.players:
            if p.finish:
                pts[p.seat] = points[p.finish - 1]
    return pts, table


def _worker(args):
    path, iters, points, seed, enforce = args
    rng = random.Random(seed)
    factory = load_agent_factory(path)
    tot = 0.0
    dist = [0.0] * 8
    decision_ns = decisions = 0
    max_ns = 0
    for _ in range(iters):
        hero = factory()
        agents = [hero] + make_zoo(rng)[:7]
        pts, table = run_table(agents, points, rng, enforce_timing=enforce)
        hp = next(p for p in table.players if p.agent is hero)
        tot += pts[hp.seat]
        if hp.finish:
            dist[hp.finish - 1] += 1
        else:  # cap survivor: approximate place by chip rank
            richer = sum(1 for p in table.players if not p.out and p.stack > hp.stack)
            dist[richer] += 1
        decision_ns += hp.decision_ns
        decisions += hp.decisions
        max_ns = max(max_ns, hp.max_ns)
    return tot, dist, decision_ns, decisions, max_ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True, help="path to your bot .py")
    ap.add_argument(
        "--iters",
        type=int,
        default=100,
        help="tables to simulate (competition uses 10,000)",
    )
    ap.add_argument(
        "--points",
        default=",".join(map(str, DEFAULT_POINTS)),
        help="comma-separated finish payouts, 1st..8th",
    )
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument(
        "--enforce-timing",
        action="store_true",
        help="fold/check on decisions exceeding 1 ms",
    )
    args = ap.parse_args()
    points = [float(x) for x in args.points.split(",")]
    assert len(points) == 8

    t0 = time.time()
    chunks = max(1, args.iters // max(args.jobs, 1))
    tasks = [
        (
            args.bot,
            min(chunks, args.iters - i * chunks),
            points,
            1234 + i,
            args.enforce_timing,
        )
        for i in range(args.jobs)
    ]
    tasks = [t for t in tasks if t[1] > 0]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]

    iters = sum(t[1] for t in tasks)
    tot = sum(r[0] for r in results)
    dist = [sum(r[1][i] for r in results) for i in range(8)]
    dns = sum(r[2] for r in results)
    ndec = sum(r[3] for r in results)
    mns = max(r[4] for r in results)

    print(f"\n=== {iters} tables in {time.time() - t0:.1f}s ===")
    print(
        f"avg points/table : {tot / iters:.3f}   (fair share = {sum(points) / 8:.2f})"
    )
    labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
    print(
        "finish dist      : "
        + "  ".join(f"{labels[i]} {d / iters:.3f}" for i, d in enumerate(dist))
    )
    if ndec:
        print(
            f"decision time    : avg {dns / ndec / 1e6:.4f} ms, "
            f"max {mns / 1e6:.2f} ms  (budget 1 ms)"
        )


if __name__ == "__main__":
    main()
