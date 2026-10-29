"""Independent Chip Model (Malmuth-Harville) for tournament payout equity.

Model: the next finisher is drawn from the remaining players with
probability proportional to chip stack; repeat until all placed.

icm_equities(stacks, payouts) -> expected payout per player.
Subset DP over 2^n masks — O(n·2^n), fine for tables up to ~14 players.
"""

from collections import defaultdict

_popcount = int.bit_count


def icm_equities(stacks, payouts):
    """Expected payout for each player still in.

    stacks: chip counts of remaining players (all > 0).
    payouts: prize for each finishing position (index 0 = 1st place).
             If shorter than len(stacks), remaining places pay 0.
    Returns list of floats summing to sum(payouts used).
    """
    n = len(stacks)
    pay = list(payouts[:n]) + [0.0] * max(0, n - len(payouts))

    # mask sums of stacks
    sum_cache = [0] * (1 << n)
    for m in range(1, 1 << n):
        lsb = m & -m
        sum_cache[m] = sum_cache[m ^ lsb] + stacks[lsb.bit_length() - 1]

    # g[mask] = probability that exactly `mask` is the unplaced set
    g = defaultdict(float)
    g[(1 << n) - 1] = 1.0
    ev = [0.0] * n
    for size in range(n, 0, -1):
        place = n - size  # finisher drawn at this size takes 'place'
        for mask in range(1 << n):
            if _popcount(mask) != size:
                continue
            gm = g[mask]
            if gm == 0.0:
                continue
            total = sum_cache[mask]
            for i in range(n):
                if not (mask >> i) & 1:
                    continue
                p = gm * stacks[i] / total
                ev[i] += p * pay[place]
                if size > 1:
                    g[mask ^ (1 << i)] += p
    return ev
