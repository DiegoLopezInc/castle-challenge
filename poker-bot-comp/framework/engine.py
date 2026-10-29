"""8-max no-limit hold'em sit-and-go engine matching competition rules.

- 2000-chip starting stacks, 10/20 blinds increasing every 25 hands
  (schedule configurable — the real one is revealed day-of)
- proper betting: min-raise = last raise increment (or BB), all-ins,
  side pots, odd chips to earliest eligible seat
- hands until one survivor or 200-hand cap; multi-bust ordering by
  stack at hand start
- agent decisions timed; over-budget / illegal / error -> check if free
  else fold (mirrors rule 3.7)

Agent protocol: agents are objects with
    act(obs) -> (kind, amount)
where kind in {'fold','check','call','raise'} and amount is the total
committed this round for a raise (raise-to), ignored otherwise.
"""

import time
from collections import deque

from .cards import eval7, new_deck

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


class Player:
    __slots__ = (
        "agent",
        "allin",
        "bet",
        "committed",
        "decision_ns",
        "decisions",
        "finish",
        "folded",
        "hole",
        "max_ns",
        "out",
        "seat",
        "stack",
    )

    def __init__(self, seat, agent, stack):
        self.seat = seat
        self.agent = agent
        self.stack = stack
        self.bet = 0  # committed this street
        self.committed = 0  # committed this hand
        self.folded = False
        self.allin = False
        self.out = False  # eliminated from tournament
        self.hole = []
        self.finish = 0  # final placement (1 = winner)
        self.decision_ns = 0  # cumulative decision time
        self.decisions = 0
        self.max_ns = 0  # slowest single decision


class Table:
    def __init__(
        self,
        agents,
        stacks=2000,
        blind_schedule=None,
        rng=None,
        time_budget_ns=1_000_000,
        enforce_timing=False,
    ):
        """agents: list of 8 agent objects (state must reset per table —
        construct fresh agents per Table, or implement .reset())."""
        import random as _r

        self.rng = rng or _r.Random()
        self.players = [Player(i, a, stacks) for i, a in enumerate(agents)]
        self.blind_schedule = blind_schedule or (
            lambda h: (10 * 2 ** (h // 25), 20 * 2 ** (h // 25))
        )
        self.time_budget_ns = time_budget_ns
        self.enforce_timing = enforce_timing
        self.button = self.rng.randrange(len(self.players))
        self.hand_num = 0
        self.finish_next = len(self.players)  # next finishing position to assign

    # ------------------------------------------------------------- helpers --

    def _alive(self):
        return [p for p in self.players if not p.out]

    def _next_alive(self, seat):
        n = len(self.players)
        for k in range(1, n + 1):
            p = self.players[(seat + k) % n]
            if not p.out:
                return p
        return None

    def _order_from(self, seat):
        """Alive seats in cyclic order starting after `seat`."""
        n = len(self.players)
        return [
            self.players[(seat + k) % n]
            for k in range(1, n + 1)
            if not self.players[(seat + k) % n].out
        ]

    def _ask(self, p, obs):
        """Call agent.act with timing + legality enforcement."""
        legal = obs["legal"]
        t0 = time.perf_counter_ns()
        try:
            act = p.agent.act(obs)
        except Exception:  # noqa: BLE001 - rules: any bot error -> fold/check
            act = None
        dt = time.perf_counter_ns() - t0
        p.decision_ns += dt
        p.decisions += 1
        p.max_ns = max(p.max_ns, dt)
        if self.enforce_timing and dt > self.time_budget_ns:
            act = None
        if not act or act[0] not in legal:
            return CHECK if CHECK in legal else FOLD, 0
        return act[0], (act[1] if len(act) > 1 else 0)

    def _obs(self, p, current_bet, last_raise, board, hand_committed, street):
        to_call = min(current_bet - p.bet, p.stack)
        legal = [FOLD]
        if to_call == 0:
            legal.append(CHECK)
        else:
            legal.append(CALL)  # covers all-in call for less than to_call
        if p.stack > to_call:  # chips left after calling -> may raise
            legal.append(RAISE)
        min_raise_to = current_bet + max(last_raise, self.bb)
        return {
            "seat": p.seat,
            "hand_num": self.hand_num,
            "street": street,
            "hole": list(p.hole),
            "board": list(board),
            "pot": hand_committed,
            "stacks": [q.stack for q in self.players],
            "committed": [q.committed for q in self.players],
            "bets": [q.bet for q in self.players],
            "to_call": to_call,
            "current_bet": current_bet,
            "min_raise_to": min_raise_to,
            "stack": p.stack,
            "bb": self.bb,
            "button": self.button,
            "alive": [not q.out for q in self.players],
            "folded": [q.folded for q in self.players],
            "legal": sorted(set(legal)),
        }

    # ---------------------------------------------------------------- hands --

    def _post(self, p, amount):
        amt = min(amount, p.stack)
        p.stack -= amt
        p.bet += amt
        p.committed += amt
        if p.stack == 0:
            p.allin = True
        return amt

    def _betting_round(self, board, street):
        """Run one street of betting. Returns False if hand over (one left)."""
        posted = max((p.bet for p in self._alive()), default=0)
        # an all-in blind for less than BB does not lower the call amount
        cur_bet = max(self.bb, posted) if street == "preflop" else posted
        last_raise = self.bb
        if street == "preflop":
            start = self._bb_seat
        else:
            start = self.button
        order = self._order_from(start)
        to_act = deque(p for p in order if p.stack > 0 and not p.allin)
        # players already at current bet (blinds) still get to act once
        while to_act:
            p = to_act.popleft()
            if p.out or p.folded or p.allin:
                continue
            # if everyone else folded/all-in and p has matched, done
            others = [
                q for q in self._alive() if q is not p and not q.folded and not q.allin
            ]
            if not others and p.bet == cur_bet:
                break
            obs = self._obs(
                p,
                cur_bet,
                last_raise,
                board,
                sum(q.committed for q in self.players),
                street,
            )
            kind, amt = self._ask(p, obs)
            if kind == FOLD:
                p.folded = True
            elif kind == CHECK:
                if p.bet != cur_bet:
                    p.folded = True  # shouldn't happen: check not in legal
            elif kind == CALL:
                need = min(cur_bet - p.bet, p.stack)
                self._post(p, need)
            elif kind == RAISE:
                target = amt  # raise-to (total this round)
                allin_to = p.bet + p.stack
                min_to = cur_bet + max(last_raise, self.bb)
                if target <= cur_bet:
                    # not really a raise -> treat as call (all-in if needed)
                    self._post(p, min(cur_bet - p.bet, p.stack))
                elif not (min_to <= target <= allin_to or target == allin_to):
                    # illegal amount -> check/fold per rules
                    if p.bet == cur_bet:
                        continue
                    p.folded = True
                    continue
                else:
                    self._post(p, target - p.bet)
                    if p.bet > cur_bet:
                        inc = p.bet - cur_bet
                        full = inc >= max(last_raise, self.bb)
                        if full:
                            last_raise = inc
                        cur_bet = p.bet
                        # a short all-in (< min raise) does not reopen
                        # action for players who already acted
                        if full:
                            to_act = deque(
                                q
                                for q in self._order_from(p.seat)
                                if q is not p
                                and not q.folded
                                and not q.allin
                                and q.stack > 0
                            )
        active = [q for q in self._alive() if not q.folded]
        for p in self._alive():
            p.bet = 0
        return len(active) > 1

    def _showdown_or_take(self, board):
        """Award pots; returns list of (seat, chips_won)."""
        contenders = [p for p in self._alive() if not p.folded]
        results = []
        if len(contenders) == 1:
            pot = sum(q.committed for q in self.players)
            contenders[0].stack += pot
            results.append((contenders[0].seat, pot))
            return results
        # side pots from committed levels
        levels = sorted({q.committed for q in self.players if q.committed > 0})
        prev = 0
        scores = {p.seat: eval7(p.hole + board) for p in contenders}
        for lv in levels:
            pot = sum(
                min(q.committed, lv) - min(q.committed, prev) for q in self.players
            )
            eligible = [p for p in contenders if p.committed >= lv]
            if pot <= 0:
                prev = lv
                continue
            best = max(scores[p.seat] for p in eligible)
            winners = [p for p in eligible if scores[p.seat] == best]
            share, rem = divmod(pot, len(winners))
            for w in winners:
                w.stack += share
                results.append((w.seat, share))
            for w in winners[:rem]:
                w.stack += 1  # odd chip(s) to earliest winner
            prev = lv
        return results

    def _bust_outs(self, start_stacks):
        """Eliminate zero-stack players; multi-bust ordered by stack at hand
        start (bigger stack = better finish)."""
        busted = [p for p in self._alive() if p.stack == 0]
        # bigger stack at hand start -> better finish: assign worst places
        # to smallest stacks first
        busted.sort(key=lambda p: start_stacks[p.seat])
        for p in busted:
            p.out = True
            p.finish = self.finish_next
            self.finish_next -= 1

    def _play_hand(self):
        self.hand_num += 1
        self.sb, self.bb = self.blind_schedule(self.hand_num - 1)
        alive = self._alive()
        start_stacks = {p.seat: p.stack for p in alive}
        for p in self.players:
            p.bet = p.committed = 0
            p.folded = p.allin = False
            p.hole = []
        deck = new_deck(self.rng)
        n = len(alive)
        if n == 2:
            # heads-up: button posts SB and acts first preflop
            sb_p = self._next_alive(self.button - 1)  # = button if alive
            self.button = sb_p.seat
            bb_p = self._next_alive(self.button)
        else:
            sb_p = self._next_alive(self.button)
            bb_p = self._next_alive(sb_p.seat)
        self._bb_seat = bb_p.seat
        self._post(sb_p, self.sb)
        self._post(bb_p, self.bb)
        # deal
        for _ in range(2):
            for q in self._order_from(sb_p.seat):
                q.hole.append(deck.pop())
        board = []
        street_seq = [("preflop", 0), ("flop", 3), ("turn", 1), ("river", 1)]
        for street, ncards in street_seq:
            board += [deck.pop() for _ in range(ncards)]
            if not self._betting_round(board, street):
                break
            left = [q for q in self._alive() if not q.folded]
            can_act = [q for q in left if not q.allin]
            if len(can_act) <= 1 and street != "river":
                continue  # deal remaining streets, no more betting
        self._showdown_or_take(board)
        self._bust_outs(start_stacks)
        # rotate button to next alive seat
        self.button = self._next_alive(self.button).seat

    def run(self, max_hands=200):
        """Play until one bot holds all chips or max_hands reached.
        Returns (finish_order, survivors) where finish_order[seat] = place
        (1 = winner) and survivors = [Player] still alive at cap."""
        while len(self._alive()) > 1 and self.hand_num < max_hands:
            self._play_hand()
        alive = self._alive()
        if len(alive) == 1:
            alive[0].finish = 1
            return [p.finish or 0 for p in self.players], []
        # hand cap: ranks assigned by ICM later; mark survivors finish=0
        return [p.finish for p in self.players], alive
