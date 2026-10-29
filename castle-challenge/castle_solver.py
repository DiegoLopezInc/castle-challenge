#!/usr/bin/env python3
"""Solver for "The Castle Challenge" — Colonel Blotto, head-to-head scoring.

100 soldiers split across 5 castles worth 1..5 points. In each matchup you win
castle i's points if a_i > b_i, split if equal. Your score is the average
points won across all opponents.

Approach
--------
For a fixed opponent *population* P, your expected score decomposes by castle:

    score(A, P) = sum_i  v_i * ( Pr[b_i < a_i] + 0.5 * Pr[b_i = a_i] )

so the best response to any population is a knapsack DP over
(5 castles) x (0..100 soldiers). Likewise, the worst-case single opponent for a
fixed allocation A is a min-convolution DP. We run a double-oracle loop:

  * candidate pool  : allocations we might play
  * adversary pool  : allocations opponents might play
  * each round, find candidate maximizing its worst case vs the adversary pool,
    then add that candidate's true worst-case opponent to the adversary pool,
    and add the best response vs the adversary mixture to the candidate pool.

Both pools are seeded with plausible human heuristics so the result is also
evaluated against a realistic field, not just self-play.
"""

import numpy as np

V = np.array([1, 2, 3, 4, 5], dtype=np.float64)   # castle values
K = 5
N = 100
RNG = np.random.default_rng(0xCA571E)


# ---------------------------------------------------------------- scoring ---

def match_score(a, b):
    """Points `a` earns in a single matchup against `b` (both len-5)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sum(V * ((a > b) + 0.5 * (a == b))))


def payoff_table(a):
    """f[i][x] = points castle i earns when opponent fields x soldiers there."""
    x = np.arange(N + 1)
    a = np.asarray(a)[:, None]
    return V[:, None] * ((x[None, :] < a) + 0.5 * (x[None, :] == a))


def worst_case(a):
    """Min over ALL legal opponent allocations of score(a, opp).

    Opponent picks b_1..b_5 summing to 100 minimizing sum_i f_i(b_i) —
    a min-convolution over the 5 castles.
    """
    f = payoff_table(a)
    dp = f[0]
    for i in range(1, K):
        new = np.full(N + 1, np.inf)
        for x in range(N + 1):
            # new[t] = min over x of dp[t-x] + f[i][x]
            seg = dp[: N + 1 - x] + f[i, x]
            new[x:] = np.minimum(new[x:], seg)
        dp = new
    return float(dp[N])


def marginals(pool):
    """pool: (M,5) allocations -> (5,101) fraction of pool fielding x at castle i."""
    m = np.zeros((K, N + 1))
    for i in range(K):
        m[i] = np.bincount(pool[:, i], minlength=N + 1) / len(pool)
    return m


def castle_gains(marg):
    """g[i][x] = expected points from castle i if we field x soldiers there."""
    x = np.arange(N + 1)
    below = np.cumsum(marg, axis=1) - marg          # Pr[b_i < x]
    return V[:, None] * (below + 0.5 * marg)         # + half on ties


def expected_score(a, marg):
    g = castle_gains(marg)
    return float(sum(g[i, a[i]] for i in range(K)))


def best_response(marg, noise=0.0):
    """Allocation maximizing expected score vs population given by marginals."""
    g = castle_gains(marg)
    if noise:
        g = g + RNG.uniform(0, noise, g.shape)
    # dp[t] = max gain over processed castles using exactly t soldiers
    NEG = -np.inf
    dp = np.full(N + 1, NEG)
    dp[0] = 0.0
    choice = []                                      # choice[i][t] = x chosen
    for i in range(K):
        new = np.full(N + 1, NEG)
        ch = np.zeros(N + 1, dtype=np.int64)
        for x in range(N + 1):
            seg = dp[: N + 1 - x] + g[i, x]
            better = seg > new[x:]
            new[x:][better] = seg[better]
            ch[x:][better] = x
        dp = new
        choice.append(ch)
    # reconstruct
    a = np.zeros(K, dtype=np.int64)
    t = N
    for i in range(K - 1, -1, -1):
        x = choice[i][t]
        a[i] = x
        t -= x
    return a


# ------------------------------------------------------- population seeds ---

def uniform_composition(m):
    """m iid uniform draws over all compositions of 100 into 5 parts."""
    # per-row uniform choice of K-1 distinct cut positions via argsort of keys
    keys = RNG.random((m, N + K - 1))
    cuts = np.sort(np.argsort(keys, axis=1)[:, : K - 1], axis=1)
    a = np.empty((m, K), dtype=np.int64)
    a[:, 0] = cuts[:, 0]
    for j in range(1, K - 1):
        a[:, j] = cuts[:, j] - cuts[:, j - 1] - 1
    a[:, K - 1] = (N + K - 2) - cuts[:, K - 2]
    return a


def noisy(base, m, spread):
    base = np.asarray(base, dtype=np.int64)
    out = np.repeat(base[None, :], m, axis=0)
    # shuffle soldiers between castles via random pairwise transfers
    for _ in range(spread):
        i = RNG.integers(0, K, m)
        j = RNG.integers(0, K, m)
        amt = RNG.integers(1, 6, m)
        ok = out[np.arange(m), i] >= amt
        out[np.arange(m)[ok], i[ok]] -= amt[ok]
        out[np.arange(m)[ok], j[ok]] += amt[ok]
    return out


def dirichlet_alloc(m, alpha):
    w = RNG.dirichlet(alpha, size=m)
    a = np.floor(w * N).astype(np.int64)
    a[:, 0] += N - a.sum(axis=1)                    # fix rounding remainder
    return a


def seed_pools():
    """Plausible opponent heuristics + generic coverage."""
    heuristics = [
        [20, 20, 20, 20, 20],     # uniform
        [7, 13, 20, 27, 33],      # proportional to value
        [0, 0, 33, 33, 34],       # abandon cheap castles
        [0, 20, 25, 27, 28],      # sacrifice castle 1
        [3, 10, 20, 30, 37],
        [1, 15, 25, 29, 30],
        [10, 15, 25, 25, 25],
        [15, 17, 20, 23, 25],
        [0, 0, 20, 40, 40],
        [25, 25, 25, 15, 10],     # inverted: grab cheap castles
        [30, 25, 20, 15, 10],
        [5, 19, 25, 26, 25],
    ]
    parts = [uniform_composition(4000)]
    parts += [noisy(h, 150, s) for h in heuristics for s in (2, 8)]
    parts += [dirichlet_alloc(1500, np.ones(K) * 2.0)]                 # random
    parts += [dirichlet_alloc(1500, V / V.sum() * 3)]                  # ~proportional
    parts += [dirichlet_alloc(1500, V[::-1] / V.sum() * 3)]            # ~inverted
    parts += [dirichlet_alloc(1500, np.array([0.5, 0.5, 1, 2, 2.5]))]  # top-heavy
    return np.vstack(parts)


def polish(a):
    """Hill-climb single-soldier transfers to maximize worst_case(a)."""
    a = np.asarray(a).copy()
    cur = worst_case(a)
    improved = True
    while improved:
        improved = False
        for i in range(K):
            for j in range(K):
                if i == j or a[i] == 0:
                    continue
                b = a.copy()
                b[i] -= 1
                b[j] += 1
                w = worst_case(b)
                if w > cur + 1e-12:
                    a, cur, improved = b, w, True
    return a, cur


# ------------------------------------------------------------- main solve ---

def score_matrix(cands, advs):
    """(C, A) matrix of matchup scores, vectorized."""
    c = cands[:, None, :]          # (C,1,5)
    a = advs[None, :, :]           # (1,A,5)
    return np.sum(V * ((c > a) + 0.5 * (c == a)), axis=2)


def solve(rounds=60):
    field = seed_pools()
    field_marg = marginals(field)

    cands = np.vstack([field[:4000], best_response(field_marg)[None, :]])
    advs = field[RNG.choice(len(field), 800, replace=False)]

    best_alloc, best_wc = None, -1.0
    for r in range(rounds):
        # 1. candidate with best worst-case vs adversary pool
        wcs = score_matrix(cands, advs).min(axis=1)
        idx = int(np.argmax(wcs))
        a_star = cands[idx]

        # 2. its true worst-case opponent (exact DP over all 4.6M allocs)
        true_wc = worst_case(a_star)
        if true_wc > best_wc:
            best_alloc, best_wc = a_star.copy(), true_wc

        # 3. grow the pools
        #    adversary: find a minimizing opponent for a_star by DP w/ backtrack
        f = payoff_table(a_star)
        dp = f[0].copy()
        choice = []
        for i in range(1, K):
            new = np.full(N + 1, np.inf)
            ch = np.zeros(N + 1, dtype=np.int64)
            for x in range(N + 1):
                seg = dp[: N + 1 - x] + f[i, x]
                better = seg < new[x:]
                new[x:][better] = seg[better]
                ch[x:][better] = x
            dp = new
            choice.append(ch)
        b = np.zeros(K, dtype=np.int64)
        t = N
        # choice[i-1] indexes castle i (i>=1); castle 0 gets remainder
        for i in range(K - 1, 0, -1):
            x = choice[i - 1][t]
            b[i] = x
            t -= x
        b[0] = t
        advs = np.vstack([advs, b[None, :]])

        #    candidate: best response to adversary mixture (small noise for diversity)
        br = best_response(marginals(advs), noise=1e-3)
        cands = np.vstack([cands, br[None, :]])

        if (r + 1) % 10 == 0:
            print(f"round {r+1:3d}  pools {len(cands)}x{len(advs)}  "
                  f"best maximin {best_wc:6.3f}  {best_alloc.tolist()}")

    # ----- final report -----------------------------------------------------
    best_alloc, best_wc = polish(best_alloc)
    br_field = best_response(field_marg)
    print("\n=== RESULT ===")
    print(f"maximin allocation: {best_alloc.tolist()}  (sum {best_alloc.sum()})")
    print(f"  worst-case score vs ANY opponent : {worst_case(best_alloc):.3f} / 15")
    print(f"  expected score vs seeded field   : {expected_score(best_alloc, field_marg):.3f} / 15")

    print(f"\nbest response to seeded field: {br_field.tolist()}  (sum {br_field.sum()})")
    print(f"  expected score vs seeded field : {expected_score(br_field, field_marg):.3f} / 15")
    print(f"  worst-case vs ANY opponent     : {worst_case(br_field):.3f} / 15")

    # sanity: a few reference allocations
    for name, a in [("uniform  ", [20, 20, 20, 20, 20]),
                    ("prop-val", [7, 13, 20, 27, 33])]:
        print(f"\n{name} {a}: field {expected_score(a, field_marg):.3f}, "
              f"worst {worst_case(a):.3f}")


def evaluate(a):
    """Score a single allocation vs the seeded field and vs its worst matchup."""
    a = np.asarray(a, dtype=np.int64)
    assert a.shape == (K,) and a.sum() == N and (a >= 0).all(), \
        "need 5 nonnegative integers summing to 100"
    field = seed_pools()
    marg = marginals(field)
    print(f"allocation {a.tolist()}")
    print(f"  expected vs seeded field : {expected_score(a, marg):.3f} / 15")
    print(f"  worst-case vs ANY opponent: {worst_case(a):.3f} / 15")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 6:
        evaluate([int(x) for x in sys.argv[1:]])
    else:
        solve()
