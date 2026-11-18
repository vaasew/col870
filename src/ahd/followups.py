"""Search decisions used only by the frozen-method follow-up pilots."""
from collections import defaultdict
import random

FOLLOWUPS = {
    'evox_reuse_rule': 'reuse_rule',
    'evox_reuse_bandit': 'reuse_bandit',
    'evox_stop_rule': 'stop_rule',
    'evox_stop_td': 'stop_td',
    'evox_lineage_diff': 'diff',
    'evox_exact_filter': 'exact',
}


class StrategyReuse:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = random.Random(seed)
        self.archive = {}
        self.stats = [[0, 0.], [0, 0.]]
        self.opportunities = 0
        self.pending = None

    def choose(self, active, remaining):
        eligible = [key for key in self.archive if key != active]
        if not eligible or remaining < 60:
            return None, len(eligible)
        self.opportunities += 1
        if self.mode == 'reuse_rule':
            reuse = self.opportunities % 2 == 1
        elif self.rng.random() < .2:
            reuse = bool(self.rng.randrange(2))
        else:
            reuse = bool(max(range(2), key=lambda a: (self.stats[a][0] == 0, self.stats[a][1])))
        self.pending = int(reuse)
        return (eligible[-1] if reuse else None), len(eligible)

    def update(self, reward):
        if self.pending is not None:
            row = self.stats[self.pending]
            row[0] += 1
            row[1] += (reward-row[1]) / row[0]
            self.pending = None


class Interruption:
    def __init__(self, mode, seed):
        self.mode = mode
        self.rng = random.Random(seed)
        self.values = defaultdict(lambda: [0., 0.])
        self.visits = defaultdict(lambda: [0, 0])
        self.pending = None

    def decide(self, state, utility):

        if self.mode == 'stop_rule':
            return not state[0] and not state[1]
        visits = self.visits[state]
        if self.rng.random() < .2:
            arm = self.rng.randrange(2)
        else:
            arm = max(range(2), key=lambda a: (visits[a] == 0, self.values[state][a]))
        self.update(utility, .9 * self.values[state][arm])
        visits[arm] += 1
        self.pending = state, arm, utility
        return bool(arm)

    def update(self, utility, continuation=0.):
        if self.pending is None:
            return
        state, arm, previous = self.pending
        value = self.values[state][arm]
        self.values[state][arm] += .2 * (utility - previous + continuation - value)
        self.pending = None
