"""Sanity tests: hand evaluator, ICM, engine invariants."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from framework.agents import CallingStation, PushFoldBot, make_zoo
from framework.cards import eval5, eval7, parse_cards
from framework.icm import icm_equities
from framework.simulate import run_table


def pc(s):
    return parse_cards(s)


def test_evaluator():
    assert eval5(pc("As Ks Qs Js Ts"))[0] == 8  # royal
    assert eval5(pc("9c 9d 9s 9h 2c"))[0] == 7  # quads
    assert eval5(pc("Kh Kd Ks 2c 2d"))[0] == 6  # boat
    assert eval5(pc("Ah Jh 8h 4h 2h"))[0] == 5  # flush
    assert eval5(pc("9s 8d 7c 6h 5s"))[0] == 4  # straight
    assert eval5(pc("As 2d 3c 4h 5s"))[1] == 3  # wheel, top=5
    assert eval5(pc("Qd Qc Qs 9h 2c"))[0] == 3  # trips
    assert eval5(pc("Kd Ks 7h 7c 2d"))[0] == 2  # two pair
    assert eval5(pc("Jd Js 8h 4c 2d"))[0] == 1  # pair
    assert eval5(pc("Ad Qs 9h 5c 3d"))[0] == 0  # high
    # kicker ordering
    assert eval5(pc("Ad Ah Ks 5c 3d")) > eval5(pc("Ad Ah Qs 5c 3d"))
    assert eval7(pc("As Ks Qs Js Ts 2d 3c")) == (8, 12)
    # 7-card picks best combo
    assert eval7(pc("Ad Kd Qd Jd Td 9d 2c"))[0] == 8
    assert eval7(pc("5c 5d 5s Kh Kd 2h 3c"))[0] == 6  # boat not trips
    print("evaluator ok")


def test_icm():
    payouts = [9, 4, 2, 0]
    eq = icm_equities([100, 100, 100, 100], payouts)
    assert abs(sum(eq) - sum(payouts)) < 1e-9
    assert all(abs(e - eq[0]) < 1e-9 for e in eq)
    eq = icm_equities([400, 200, 100, 100], payouts)
    assert eq[0] > eq[1] > eq[2] == eq[3]
    # winner-take-all -> equity proportional to stacks
    eq = icm_equities([300, 100], [9])
    assert abs(eq[0] - 6.75) < 1e-9 and abs(eq[1] - 2.25) < 1e-9
    print("icm ok")


def test_engine_invariants():
    rng = random.Random(0)
    for t in range(15):
        agents = [CallingStation() for _ in range(4)] + [
            PushFoldBot(rng=rng) for _ in range(4)
        ]
        pts = [9, 7, 5, 4, 3, 2, 1, 0]
        scores, table = run_table(agents, pts, rng, max_hands=200)
        # chips conserved: 8 players x 2000 = 16000 distributed to winners
        total = sum(p.stack for p in table.players)
        assert total == 16000, f"table {t}: chips={total}"
        # points fully allocated
        assert abs(sum(scores) - sum(pts)) < 1e-6, scores
        # all players accounted for
        finishes = [p.finish for p in table.players if p.finish]
        survivors = [p for p in table.players if not p.finish]
        assert len(finishes) + len(survivors) == 8
        assert len(set(finishes)) == len(finishes)  # distinct places
    print("engine invariants ok")


def test_speed():
    import time

    rng = random.Random(1)
    t0 = time.time()
    n = 3
    for _ in range(n):
        run_table(make_zoo(rng), [9, 7, 5, 4, 3, 2, 1, 0], rng)
    dt = time.time() - t0
    print(f"speed: {dt / n:.2f}s/table  ({n / dt:.1f} tables/s)")


if __name__ == "__main__":
    test_evaluator()
    test_icm()
    test_engine_invariants()
    test_speed()
