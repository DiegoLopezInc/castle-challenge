#!/usr/bin/env python3
"""Decision analysis for the Castle Challenge under field uncertainty.

Builds several plausible field archetypes, computes each candidate's expected
score in every scenario, and reports:
  * Bayes-optimal pick (equal-weight mixture of archetypes)
  * minimax-regret pick: regret(a, F) = score(BR(F), F) - score(a, F)
  * best response to a self-play (fictitious-play) converged field
"""

import numpy as np
from castle_solver import (V, K, N, RNG, marginals, castle_gains,
                           expected_score, best_response,
                           uniform_composition, noisy, dirichlet_alloc)


def archetypes():
    """Named plausible field distributions -> (M,5) pools."""
    balanced = noisy([20, 20, 20, 20, 20], 600, 6)
    propval = noisy([7, 13, 20, 27, 33], 600, 6)
    topheavy = np.vstack([
        noisy([2, 0, 0, 49, 49], 300, 4),
        noisy([0, 2, 2, 48, 48], 300, 4),
        dirichlet_alloc(600, np.array([0.4, 0.4, 0.6, 1.6, 2.0])),
    ])
    cheapgrab = np.vstack([
        noisy([30, 25, 20, 15, 10], 400, 5),
        dirichlet_alloc(800, V[::-1] / V.sum() * 3),
    ])
    random_ = uniform_composition(1500)
    return {
        "balanced": balanced,
        "prop-val": propval,
        "top-heavy": topheavy,
        "cheap-grab": cheapgrab,
        "uniform-random": random_,
    }


def fictitious_play_field(rounds=300):
    """Self-play field: start random, repeatedly add BR to current mixture."""
    pool = uniform_composition(2000)
    for _ in range(rounds):
        pool = np.vstack([pool, best_response(marginals(pool), noise=1e-3)[None]])
    return pool


def candidates():
    """Candidate allocations to compare (BRs added later per-scenario)."""
    return {
        "BR(seed-field)": [0, 2, 26, 34, 38],
        "maximin":        [0, 2, 2, 48, 48],
        "uniform":        [20, 20, 20, 20, 20],
        "prop-val":       [7, 13, 20, 27, 33],
        "exploit":        [2, 0, 0, 49, 49],
        "cheap-grab":     [25, 25, 20, 18, 12],
    }


def main():
    fields = archetypes()
    fields["self-play"] = fictitious_play_field()
    margs = {name: marginals(f) for name, f in fields.items()}

    # per-scenario best responses become additional candidates
    cands = candidates()
    for name, m in margs.items():
        cands[f"BR({name})"] = best_response(m)

    names = list(cands)
    scen = list(margs)
    scores = np.zeros((len(names), len(scen)))
    for i, cn in enumerate(names):
        a = np.asarray(cands[cn])
        for j, sn in enumerate(scen):
            scores[i, j] = expected_score(a, margs[sn])

    # add the self-play BR candidate's own row already included; now report
    br_scores = scores.max(axis=0)                     # best achievable per scenario
    regret = br_scores[None, :] - scores

    hdr = f"{'candidate':<16}" + "".join(f"{s:>16}" for s in scen)
    print(hdr + f"{'Bayes':>8}{'maxRegret':>10}")
    print("-" * len(hdr) + "-" * 18)
    for i, cn in enumerate(names):
        row = "".join(f"{scores[i, j]:>16.2f}" for j in range(len(scen)))
        print(f"{cn:<16}{row}{scores[i].mean():>8.2f}{regret[i].max():>10.2f}")

    print("\nBest achievable per scenario:",
          {s: round(float(v), 2) for s, v in zip(scen, br_scores)})

    bayes = int(np.argmax(scores.mean(axis=1)))
    mmr = int(np.argmin(regret.max(axis=1)))
    print(f"\nBayes-optimal (equal scenario weights): {names[bayes]} "
          f"{np.asarray(cands[names[bayes]]).tolist()}")
    print(f"Minimax-regret pick:                    {names[mmr]} "
          f"{np.asarray(cands[names[mmr]]).tolist()}")

    sp = np.asarray(cands["BR(self-play)"])
    print(f"\nSelf-play converged BR: {sp.tolist()}  "
          f"scores {dict(zip(scen, np.round(scores[names.index('BR(self-play)')], 2)))}")


if __name__ == "__main__":
    main()
