"""Agent protocol and opponent zoo for the competition simulator.

Protocol (placeholder — the real Agent interface ships day-of):

    class MyAgent:
        def act(self, obs) -> (kind, amount)

obs keys: seat, hand_num, street, hole, board, pot, stacks, committed,
bets, to_call, current_bet, min_raise_to, stack, bb, button, alive,
folded, legal

kind in {'fold','check','call','raise'}; amount = raise-to total this
round (only for 'raise').

Zoo bots are deliberately diverse archetypes, not GTO — they stand in for
the mixed field of ~student-built bots you'll actually face.
"""

import random

from .cards import eval7

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


def hand_rank_preflop(hole):
    """Rough preflop strength 0..1: pairs > big aces > suited broadways ..."""
    r1, r2 = sorted((c // 4 for c in hole), reverse=True)
    s1, s2 = hole[0] % 4, hole[1] % 4
    if r1 == r2:
        return 0.55 + r1 / 13 * 0.45
    base = (r1 + r2) / 24
    if s1 == s2:
        base += 0.07
    if abs(r1 - r2) <= 2:
        base += 0.05
    if r1 == 12:
        base += 0.10
    return min(base, 0.95)


def made_hand_strength(hole, board):
    """0..1 postflop strength bucket from the made hand (draws ignored)."""
    if len(board) < 3:
        return hand_rank_preflop(hole)
    cat = eval7(hole + board)[0]
    return min(cat / 8 + 0.1, 1.0)  # cat 0->.1 ... SF->1.0


class Agent:
    def reset(self):
        """Called at table start (rule 3.8: state resets each table)."""

    def act(self, obs):
        raise NotImplementedError


class RandomBot(Agent):
    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def act(self, obs):
        if "raise" in obs["legal"] and self.rng.random() < 0.1:
            return RAISE, min(
                obs["min_raise_to"], obs["stack"] + obs["bets"][obs["seat"]]
            )
        if "call" in obs["legal"]:
            return CALL, 0
        return (CHECK, 0) if "check" in obs["legal"] else (FOLD, 0)


class CallingStation(Agent):
    def act(self, obs):
        if "call" in obs["legal"]:
            return CALL, 0
        return (CHECK, 0) if "check" in obs["legal"] else (FOLD, 0)


class PushFoldBot(Agent):
    """ICM-flavored push/fold: shove decent hands when short, fold otherwise.

    This approximates near-optimal SNG endgame play — arguably the strongest
    realistic opponent in a student field."""

    def __init__(self, shove_bb=12, rng=None):
        self.shove_bb = shove_bb
        self.rng = rng or random.Random()

    def act(self, obs):
        s = hand_rank_preflop(obs["hole"])
        bb = obs["bb"]
        stack_bb = obs["stack"] / bb
        if stack_bb <= self.shove_bb:
            threshold = 0.25 + 0.05 * stack_bb / self.shove_bb
            if s >= threshold and "raise" in obs["legal"]:
                return RAISE, obs["stack"] + obs["bets"][obs["seat"]]
        if obs["to_call"] == 0:
            return CHECK, 0
        if obs["to_call"] <= bb and s > 0.5:
            return CALL, 0
        return FOLD, 0


class HeuristicBot(Agent):
    """Simple TAG: raise good preflop hands, cbet flops, fold weak to pressure."""

    def __init__(self, aggression=0.7, tight=0.42, rng=None):
        self.aggression = aggression
        self.tight = tight
        self.rng = rng or random.Random()

    def act(self, obs):
        legal = obs["legal"]
        to_call = obs["to_call"]
        pot = obs["pot"]
        if obs["street"] == "preflop":
            s = hand_rank_preflop(obs["hole"])
            if "raise" in legal and s >= self.tight + 0.25:
                return RAISE, obs["min_raise_to"]
            if s >= self.tight and to_call <= 3 * obs["bb"] and "call" in legal:
                return CALL, 0
            return (CHECK, 0) if "check" in legal else (FOLD, 0)
        s = made_hand_strength(obs["hole"], obs["board"])
        pot_frac = to_call / max(pot, 1)
        if "raise" in legal and s >= 0.5 and self.rng.random() < self.aggression:
            target = obs["current_bet"] + max(int(pot * 0.6), obs["bb"])
            return RAISE, max(target, obs["min_raise_to"])
        if "call" in legal and s >= 0.35 + pot_frac * 0.5 and to_call <= pot:
            return CALL, 0
        if "check" in legal:
            return CHECK, 0
        return FOLD, 0


class TrappyBot(HeuristicBot):
    """LAG-ish: calls wider, bluffs sometimes."""

    def __init__(self, rng=None):
        super().__init__(aggression=0.5, tight=0.30, rng=rng)

    def act(self, obs):
        if (
            obs["street"] != "preflop"
            and "raise" in obs["legal"]
            and self.rng.random() < 0.08
        ):
            return RAISE, obs["min_raise_to"]  # light bluff
        return super().act(obs)


ZOO = {
    "random": RandomBot,
    "station": CallingStation,
    "pushfold": PushFoldBot,
    "tag": HeuristicBot,
    "lag": TrappyBot,
}


def make_zoo(rng=None):
    """A plausible field: mix of archetypes."""
    return [
        CallingStation(),
        HeuristicBot(rng=rng),
        HeuristicBot(rng=rng),
        TrappyBot(rng=rng),
        PushFoldBot(rng=rng),
        PushFoldBot(rng=rng),
        RandomBot(rng=rng),
        HeuristicBot(rng=rng),
    ]
