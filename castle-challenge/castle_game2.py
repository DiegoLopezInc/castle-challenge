#!/usr/bin/env python3
"""Solver for Game 2: four-way battle.

Random group of 4 applicants. Each castle goes to whoever fields the most
soldiers there; ties split the castle's points among the tied players.
Players are ranked by total castle points; payouts 9/4/2/0, with tied places
splitting their combined payouts. Objective: expected payout over random groups.

Method: vectorized Monte Carlo evaluator + evolutionary search over allocations,
with common random numbers for stable selection. Evaluated against a field
prior (default: mostly non-naive, matching the Game-1 discussion).
"""

import numpy as np
from castle_solver import (V, K, N, RNG, marginals, expected_score,
                           best_response, uniform_composition, noisy,
                           dirichlet_alloc, seed_pools, worst_case)
from scenarios import fictitious_play_field

PAYOUTS = np.array([9.0, 4.0, 2.0, 0.0])
# points scaled by 12 keep all k-way splits (k<=4) integer-exact
SCALE = 12

# payout lookup: PAY[i][j] = our payout when i opponents beat us and j tie us
PAY = np.zeros((4, 4))
for i in range(4):
    for j in range(4 - i):
        PAY[i, j] = PAYOUTS[i:i + j + 1].mean()


def expected_payout(a, field, groups):
    """Monte Carlo expected payout of allocation `a` vs `field` pool.

    groups: (G,3) opponent indices into `field` (common random numbers).
    Returns (mean payout, finish-dist array over 4 places incl. ties share).
    """
    G = len(groups)
    opps = field[groups]                                # (G,3,5)
    us = np.broadcast_to(np.asarray(a), (G, 1, K))
    allocs = np.concatenate([us, opps], axis=1)         # (G,4,5)

    mx = allocs.max(axis=1, keepdims=True)              # per-castle max
    atmax = allocs == mx                                # (G,4,5)
    n_tied = atmax.sum(axis=1)                          # (G,5) in 1..4
    pts = (atmax * (SCALE * V[None, None, :]) // n_tied[:, None, :]
           ).sum(axis=2)                                # (G,4) scaled pts

    ours = pts[:, 0:1]
    above = (pts[:, 1:] > ours).sum(axis=1)
    equal = (pts[:, 1:] == ours).sum(axis=1)
    pay = PAY[above, equal]
    return pay.mean(), dist_from(above, equal, G)


def dist_from(above, equal, G):
    """Expected finish-place distribution incl. tie-splitting."""
    dist = np.zeros(4)
    for i in range(4):
        for j in range(4 - i):
            cnt = np.sum((above == i) & (equal == j))
            if cnt:
                dist[i:i + j + 1] += cnt / (j + 1)
    return dist / G


# ------------------------------------------------------------- field prior ---

def field_prior():
    """Mostly non-naive field: sophisticated cloud + self-play pool + naive tail."""
    soph = np.array([1, 3, 13, 38, 45])
    oneup = np.array([1, 0, 14, 39, 46])
    sp = fictitious_play_field(300)
    naive = seed_pools()
    field = np.vstack([
        noisy(soph, 1200, 6),          # converged-ish sophisticated play
        noisy(oneup, 800, 5),          # people one-upping the consensus
        sp[-1500:],                    # self-play converged population
        naive[RNG.choice(len(naive), 2500, replace=False)],  # naive tail
    ])
    return field


# ---------------------------------------------------------------- evolution ---

def mutate(a, n_moves=None):
    a = a.copy()
    for _ in range(n_moves or RNG.integers(1, 5)):
        i, j = RNG.integers(0, K, 2)
        d = min(a[i], RNG.integers(1, 8))
        a[i] -= d
        a[j] += d
    return a


def evolve(field, pop_size=192, gens=45, n_groups=15000):
    groups = RNG.integers(0, len(field), size=(n_groups, 3))   # CRN
    pop = [np.array(x) for x in
           [[1, 0, 14, 39, 46], [0, 2, 2, 48, 48], [1, 3, 13, 38, 45],
            [0, 2, 26, 34, 38], [0, 0, 30, 35, 35], [0, 0, 0, 50, 50],
            [0, 0, 20, 40, 40], [20, 20, 20, 20, 20]]]
    pop += [mutate(pop[RNG.integers(len(pop))]) for _ in range(pop_size - len(pop))]

    for g in range(gens):
        fit = np.array([expected_payout(a, field, groups)[0] for a in pop])
        order = np.argsort(-fit)
        pop = [pop[i] for i in order]
        if g % 10 == 9 or g == 0:
            print(f"gen {g+1:3d}  best {fit[order[0]]:.3f}  {pop[0].tolist()}")
        elite = pop[:pop_size // 3]
        pop = elite + [mutate(elite[RNG.integers(len(elite))])
                       for _ in range(pop_size - len(elite))]
    return pop[0], fit[order[0]]


def main():
    field = field_prior()
    best, fit = evolve(field)

    # fresh eval on independent draws, incl. finish distribution
    groups = RNG.integers(0, len(field), size=(200_000, 3))
    refs = {
        "evolved": best,
        "one-up [1,0,14,39,46]": np.array([1, 0, 14, 39, 46]),
        "maximin [0,2,2,48,48]": np.array([0, 2, 2, 48, 48]),
        "soph [1,3,13,38,45]": np.array([1, 3, 13, 38, 45]),
        "uniform": np.array([20, 20, 20, 20, 20]),
        "prop-val": np.array([7, 13, 20, 27, 33]),
    }
    print(f"\n{'candidate':<24}{'E[payout]':>10}  1st   2nd   3rd   4th")
    for name, a in refs.items():
        p, dist = expected_payout(np.asarray(a), field, groups)
        print(f"{name:<24}{p:>10.3f}  " +
              "  ".join(f"{d:.3f}" for d in dist))


if __name__ == "__main__":
    main()
