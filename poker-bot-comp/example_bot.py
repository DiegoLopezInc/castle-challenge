"""Example bot — safe-baseline template for the day-of Agent interface.

Strategy: conservative push/fold. Shove premium hands when short-stacked,
never bust to a timeout or illegal action. This is a floor to beat, not a
contender — but it finishes respectably because most losses in SNGs come
from mistakes this bot simply doesn't make.
"""

from framework.agents import (
    CALL,
    CHECK,
    FOLD,
    RAISE,
    Agent,
    hand_rank_preflop,
    made_hand_strength,
)


class MyBot(Agent):
    def act(self, obs):
        hole, board = obs["hole"], obs["board"]
        stack, bb = obs["stack"], obs["bb"]
        to_call = obs["to_call"]
        legal = obs["legal"]

        if not board:  # preflop
            s = hand_rank_preflop(hole)
            if stack <= 10 * bb and s >= 0.60 and "raise" in legal:
                return RAISE, stack + obs["bets"][obs["seat"]]  # shove
            if s >= 0.72 and "raise" in legal and to_call <= 3 * bb:
                return RAISE, obs["min_raise_to"]
            if s >= 0.55 and to_call <= 2 * bb and "call" in legal:
                return CALL, 0
            return (CHECK, 0) if "check" in legal else (FOLD, 0)

        s = made_hand_strength(hole, board)
        if to_call == 0:
            if s >= 0.6 and "raise" in legal:
                return RAISE, max(
                    obs["min_raise_to"], int(obs["pot"] * 0.5) + obs["current_bet"]
                )
            return CHECK, 0
        # facing a bet: continue with a real hand at a fair price
        if s >= 0.5 and to_call <= max(obs["pot"] * 0.5, bb) and "call" in legal:
            return CALL, 0
        return FOLD, 0


def make_agent():
    return MyBot()
