"""Cards, deck, and a fast pure-Python 7-card hold'em evaluator.

Cards are ints 0..51: rank = c // 4 (0='2' .. 12='A'), suit = c % 4.
Scores are tuples (category, *tiebreakers) — higher is better, directly
comparable. Categories: 8 SF, 7 quads, 6 full house, 5 flush, 4 straight,
3 trips, 2 two pair, 1 pair, 0 high card.
"""

import random
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_str(c):
    return RANKS[c // 4] + SUITS[c % 4]


def parse_cards(s):
    """'As Kd' -> [48, 27]"""
    return [
        RANKS.index(t[0].upper()) * 4 + SUITS.index(t[1].lower()) for t in s.split()
    ]


def new_deck(rng=random):
    d = list(range(52))
    rng.shuffle(d)
    return d


def eval5(cards):
    """Score a 5-card hand. cards: iterable of 5 ints."""
    ranks = [0] * 13
    suits = [0] * 4
    for c in cards:
        ranks[c // 4] += 1
        suits[c % 4] += 1
    flush = any(s == 5 for s in suits)
    # straight: longest run of present ranks, record highest top card
    run = 0
    top = -1
    for r in range(13):
        if ranks[r]:
            run += 1
            if run >= 5:
                top = r
        else:
            run = 0
    if top < 0 and ranks[12] and all(ranks[r] for r in range(4)):
        top = 3  # wheel A2345, high card = 5
    straight = top >= 0
    # counts by multiplicity
    quad = trip = -1
    pairs = []
    singles = []
    for r in range(12, -1, -1):
        n = ranks[r]
        if n == 4:
            quad = r
        elif n == 3:
            trip = r
        elif n == 2:
            pairs.append(r)
        elif n == 1:
            singles.append(r)
    if straight and flush:
        return (8, top)
    if quad >= 0:
        return (7, quad, max(pairs + singles))
    if trip >= 0 and pairs:
        return (6, trip, pairs[0])
    if flush:
        hi = tuple(r for r in range(12, -1, -1) if ranks[r])[:5]
        return (5, *hi)
    if straight:
        return (4, top)
    if trip >= 0:
        return (3, trip, *singles[:2])
    if len(pairs) == 2:
        return (2, pairs[0], pairs[1], singles[0])
    if len(pairs) == 1:
        return (1, pairs[0], *singles[:3])
    return (0, *singles[:5])


def eval7(cards):
    """Best 5-card score over a 7-card hand (also works for 5-6 cards)."""
    if len(cards) == 5:
        return eval5(cards)
    best = None
    for combo in combinations(cards, 5):
        s = eval5(combo)
        if best is None or s > best:
            best = s
    return best


def hand_name(score):
    names = [
        "high card",
        "pair",
        "two pair",
        "trips",
        "straight",
        "flush",
        "full house",
        "quads",
        "straight flush",
    ]
    return names[score[0]]
