"""Opt-in controls around the pinned EvoX candidate sampling boundary."""
from collections import defaultdict, deque
import hashlib
import random

from .audit import event
from .protocol import score_profile

SEARCH_LEVERS = {'evox_lineage_memory': 'lineage_immediate', 'evox_lineage_reward': 'lineage_incumbent'}


class SearchLever:
    def __init__(self, mode, seed, folder, *, specialist_fraction=0.):
        self.mode = mode
        self.rng = random.Random(seed)
        self.folder = folder
        self.specialist_fraction = specialist_fraction
        self.specialist_rng = random.Random(seed + 91_271)
        self.outcomes = deque(maxlen=8)
        self.values = defaultdict(lambda: [0, 0.])
        self.families = {}
        self.duplicates = defaultdict(int)
        self.cool_until = {}
        self.last_gain = 0
        self.lineages = defaultdict(lambda: [0, 0.])
        self.ancestry = {}
        self.incumbent = None
        self.choice = None
        self.duplicate = False

    def sample(self, controller, native, iteration, *args, **kwargs):
        raw, context = native(*args, **kwargs)
        label, parent = next(iter(raw.items())) if isinstance(raw, dict) else ('', raw)
        original = parent
        programs = controller.database.programs
        family = self.families.setdefault(parent.id, parent.id if self.mode.startswith('lineage_')
                                          else parent.parent_id or parent.id)
        controller._index_scores()

        def record(program):
            return controller.score_records.get(hashlib.sha256(program.solution.encode()).hexdigest())

        if self.mode.startswith('lineage_'):
            eligible = {}
            scores = [row['score'] for row in controller.score_records.values() if row['valid']]
            self.incumbent = max(scores, default=None)
            for candidate in programs.values():
                branch = self.families.get(candidate.id, candidate.id)
                row = record(candidate)
                if branch == family or not row or not row['valid']:
                    continue
                if branch not in eligible or row['score'] > eligible[branch][0]:
                    eligible[branch] = row['score'], candidate
            if eligible:
                if self.rng.random() < .2:
                    branch = self.rng.choice([family, *eligible])
                else:
                    branch = max([family, *eligible], key=lambda key:
                                 (self.lineages[key][0] == 0,
                                  self.lineages[key][1] / max(1, self.lineages[key][0])))
                if branch != family:
                    parent = eligible[branch][1]
        elif self.mode == 'tail':
            before = record(parent)
            profile = score_profile(before['result']) if before and before['valid'] else ()
            if len(profile) > 1:
                def lower_tail(values):
                    tail = sorted(values)[:max(1, len(values) // 3)]
                    return sum(tail) / len(tail)

                cutoff = before['score'] - .02 * max(abs(before['score']), 1e-9)
                best_tail = lower_tail(profile)
                for candidate in programs.values():
                    row = record(candidate)
                    if not row or not row['valid'] or row['score'] < cutoff:
                        continue
                    scores = score_profile(row['result'])
                    if len(scores) == len(profile) and lower_tail(scores) > best_tail:
                        parent, best_tail = candidate, lower_tail(scores)
        elif self.mode == 'cooldown' and iteration < self.cool_until.get(family, 0):
            pool = [p for p in programs.values()
                    if iteration >= self.cool_until.get(self.families.get(p.id, p.parent_id or p.id), 0)
                    and (row := record(p)) and row['valid']]
            if pool:
                parent = self.rng.choice(pool)
        state = None
        arm = None
        if self.mode in ('mix', 'bandit'):
            labels = (controller.database.REFINE_LABEL, controller.database.DIVERGE_LABEL)
            state = (iteration - self.last_gain >= 8,
                     bool(self.outcomes) and 2 * sum(self.outcomes) >= len(self.outcomes))
            if self.mode == 'mix' or self.rng.random() < .15:
                arm = self.rng.randrange(2)
            else:
                arm = max(range(2), key=lambda i: (self.values[state, i][0] == 0,
                                                    self.values[state, i][1]))
            label = labels[arm]

        if self.specialist_fraction and self.specialist_rng.random() < self.specialist_fraction:
            specialists = {}
            for candidate in programs.values():
                row = record(candidate)
                if not row or not row['valid'] or not row['result'].get('causality_passed'):
                    continue
                for case, score in enumerate(score_profile(row['result'])):
                    rank = (score, row['score'])
                    if case not in specialists or rank > specialists[case][0]:
                        specialists[case] = rank, candidate
            if specialists:
                case = self.specialist_rng.choice(sorted(specialists))
                lineage_parent = parent.id
                parent = specialists[case][1]
                event(self.folder, 'specialist_parent', iteration=iteration, training_case=case,
                      lineage_parent=lineage_parent, selected_parent=parent.id,
                      changed=parent.id != lineage_parent)
            else:
                event(self.folder, 'specialist_parent_unavailable', iteration=iteration)

        if parent.id != original.id:
            if isinstance(context, dict):
                context = {key: [p for p in rows if p.id != parent.id] for key, rows in context.items()}
            else:
                context = [p for p in context if p.id != parent.id]
        family = self.families.setdefault(parent.id, parent.id if self.mode.startswith('lineage_')
                                          else parent.parent_id or parent.id)
        self.choice = parent, record(parent), family, state, arm
        event(self.folder, 'lever_choice', iteration=iteration, mode=self.mode,
              native_parent=original.id, parent=parent.id, family=family, state=state, arm=arm,
              operator=label)
        return ({label: parent} if isinstance(raw, dict) or arm is not None else parent), context

    def observe(self, controller, result, iteration, elapsed):
        if self.choice is None:
            return
        parent, before, family, state, arm = self.choice
        child = result.child_program_dict or {}
        source = child.get('solution')
        controller._index_scores()
        after = controller.score_records.get(hashlib.sha256(source.encode()).hexdigest()) if source else None
        valid = bool(after and after['valid'] and not result.error and not self.duplicate)
        gain = after['score'] - before['score'] if valid and before and before['valid'] else 0.
        reward = max(0., gain)
        breakthrough = 0.
        if self.mode == 'lineage_incumbent' and reward and self.incumbent is not None:
            breakthrough = min(reward, max(0., after['score'] - self.incumbent))
            reward = .25 * (reward - breakthrough) + breakthrough
        self.outcomes.append(valid)
        if reward:
            self.last_gain = iteration
        if self.mode.startswith('lineage_'):
            stats = self.lineages[family]
            stats[0] += 1
            stats[1] += reward
            if child.get('id') and valid:
                child_id = child['id']
                self.families[child_id] = family if parent.parent_id else child_id
                ancestors = self.ancestry.get(parent.id, ())
                self.ancestry[child_id] = (family, *ancestors[:2])
                bonus = max(0., after['score'] - self.incumbent) if self.incumbent is not None else 0.
                if self.mode == 'lineage_delayed' and bonus and ancestors:
                    for branch in dict.fromkeys(ancestors):
                        self.lineages[branch][1] += bonus
                    event(self.folder, 'lineage_credit', iteration=iteration,
                          branches=list(dict.fromkeys(ancestors)), bonus=bonus)
        elif child.get('id'):
            self.families[child['id']] = family
        if self.mode == 'bandit' and arm is not None:
            count, mean = self.values[state, arm]
            self.values[state, arm] = [count + 1, mean + (reward - mean) / (count + 1)]
        if self.mode == 'cooldown':
            self.duplicates[family] = self.duplicates[family] + 1 if self.duplicate else 0
            if self.duplicates[family] >= 2:
                self.cool_until[family] = iteration + 5
                self.duplicates[family] = 0
        event(self.folder, 'lever_outcome', iteration=iteration, mode=self.mode,
              family=family, valid=valid, duplicate=self.duplicate, gain=gain, reward=reward,
              breakthrough=breakthrough, elapsed=elapsed, cooldown_until=self.cool_until.get(family), state=state, arm=arm)
