"""Opt-in EvoX hooks. The bare launch still uses the upstream entry point."""
import ast
from difflib import unified_diff
import hashlib
import json
import os
from collections import defaultdict
import random
from pathlib import Path
import time

import numpy as np
from skydiscover.main import main
from skydiscover.optimize.search.evox.controller import CoEvolutionController
from skydiscover.optimize.search.utils.discovery_utils import SerializableResult
from skydiscover.optimize.search.route import register_controller
from skydiscover.optimize.utils.code_utils import parse_full_rewrite

from .audit import event
from .addons import Pacing, critique, measured_feedback
from .followups import Interruption, StrategyReuse
from .memory import changed_functions, retrieve
from .search_levers import SearchLever
from .prompt_tuning import PromptTuner
from .protocol import EDITABLE_ENTRIES, score_profile, task_path


START = '# EVOLVE-BLOCK-START'
END = '# EVOLVE-BLOCK-END'


def assemble_block(initial, response, task):
    prefix, _, tail = initial.partition(START)
    _, marker, suffix = tail.partition(END)
    if not tail or not marker:
        raise ValueError('Starter has no editable block')
    body = parse_full_rewrite(response, 'python').strip()
    if START in body or END in body:
        if body.count(START) != 1 or body.count(END) != 1 or body.index(START) > body.index(END):
            raise ValueError('Invalid EVOLVE markers')
        body = body.split(START, 1)[1].split(END, 1)[0].strip()
    tree = ast.parse(body)
    entry = EDITABLE_ENTRIES[task]
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions.count(entry) != 1:
        raise ValueError(f'Editable block must define {entry} exactly once')
    fixed_functions = {node.name for node in ast.parse(prefix + suffix).body
                       if isinstance(node, ast.FunctionDef)}
    if fixed_functions.intersection(functions):
        raise ValueError('Leave fixed wrapper functions to the starter')
    assembled = prefix + START + '\n' + body + '\n' + END + suffix
    ast.parse(assembled)
    return assembled


class MetaPolicy:
    """Linear values for pairs of meta-search actions and commitment lengths."""

    def __init__(self, kind, seed, lengths=(4,)):
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.options = [(meta, length) for length in lengths for meta in (0, 1)]
        self.weights = np.zeros((len(self.options), 8))
        self.counts = [0] * len(self.options)

    def feasible(self, meta_allowed):
        return [i for i, (meta, _) in enumerate(self.options) if meta_allowed or not meta]

    def choose(self, features, meta_allowed, rule_due, meta_penalty=0.):
        feasible = self.feasible(meta_allowed)
        if len(feasible) == 1:
            action = 0
        elif self.kind == 'rule':
            action = int(rule_due)
        else:
            untried = [i for i in feasible if self.counts[i] == 0]
            if len(self.options) > 2:
                warmup = [i for i in untried if self.options[i][1] == 4]
                untried = warmup or untried
            values = self.weights @ features
            if meta_penalty:
                values[1] -= meta_penalty
            if untried:
                action = int(self.rng.choice(untried))
            elif self.rng.random() < .2:
                action = int(self.rng.integers(2) if len(self.options) == 2 else self.rng.choice(feasible))
            else:
                best = max(values[i] for i in feasible)
                action = int(self.rng.choice([i for i in feasible if values[i] == best]))
        self.counts[action] += 1
        return action

    def update(self, action, features, reward, next_features, meta_allowed, terminal):
        target = reward
        if self.kind == 'td' and not terminal:
            target += max(self.weights[self.feasible(meta_allowed)] @ next_features)
        prediction = self.weights[action] @ features
        if self.kind != 'rule':
            self.weights[action] += .15 * (target-prediction) * features / max(1., features @ features)
        return float(target), float(prediction)


class ExperimentalEvoX(CoEvolutionController):
    """Use native search and override only candidate assembly or the meta trigger."""

    def __init__(self, controller_input):
        super().__init__(controller_input)
        self.guard = Path(os.environ['AHD_GUARD_DIR'])
        self.folder = self.guard.parent
        config = json.loads((self.folder/'manifest.json').read_text())['config']
        self.course = config
        self.addon = config.get('addon')
        self.followup = config.get('followup')
        self.reuse = (StrategyReuse(self.followup, config['seed'])
                      if self.followup in ('reuse_rule', 'reuse_bandit') else None)
        self.interruption = (Interruption(self.followup, config['seed'])
                             if self.followup in ('stop_rule', 'stop_td') else None)
        self.pacing = Pacing(self.addon) if self.addon in ('pacing', 'cost') else None
        self.pending_feedback = None
        self.critique_calls = 0
        self.iteration = 0
        self.initial = (task_path(config['task'])/'initial_program.py').read_text()
        self.policy = (MetaPolicy(config['scheduler'], config['seed'], config.get('commitment_lengths', [4]))
                       if config['scheduler'] != 'native' else None)
        self.boundary_ticks = 0
        self.next_boundary = 4
        self.rule_due = False
        self.previous = None
        self.meta_seconds = 60.
        self.initialization_attempted = False
        self.strategy_since = time.time()
        self.last_improvement = self.strategy_since
        self.last_score = 0.
        self.lever = (SearchLever(config['search_lever'], config['seed'], self.folder,
                                  specialist_fraction=config.get('specialist_fraction', 0.))
                      if config.get('search_lever') else None)
        self.memory_gate = config.get('memory_gate')
        self.gate_rng = random.Random(config['seed'])
        self.gate_values = defaultdict(lambda: [0, 0.])
        self.gate_choice = None
        self.memory_mode = config.get('memory_mode')
        self.history = []
        self.score_records = {}
        self.indexed_slots = 0
        self.memory_query = None
        self.tuner = (PromptTuner(self.folder, self.config.llm.api_base, config['prompt_mode'],
                                  (task_path(config['task'])/'prompt.txt').read_text(), self.initial)
                      if config.get('prompt_mode') else None)

    def _index_scores(self):
        used = json.loads((self.guard/'ledger.json').read_text())['used']
        for index in range(self.indexed_slots, used):
            path = self.guard/f'record_{index:06d}.json'
            if path.exists():
                record = json.loads(path.read_text())
                self.score_records[record['hash']] = record
        self.indexed_slots = used

    def _build_prompt(self, current_program, context_programs, failed_attempts):
        prompt = super()._build_prompt(current_program, context_programs, failed_attempts)
        if self.tuner and self.tuner.instruction:
            guidance = ('\n<<<AHD_OPTIONAL:mutation_guidance>>>\nMutation guidance:\n'
                        + self.tuner.instruction + '\n<<<AHD_END_OPTIONAL>>>\n')
            prompt = {**prompt, 'user': prompt['user'] + guidance}
        if not self.memory_mode:
            return prompt
        if isinstance(current_program, dict):
            operator, parent = next(iter(current_program.items()))
        else:
            operator, parent = '', current_program
        self._index_scores()
        parent_record = self.score_records.get(hashlib.sha256(parent.solution.encode()).hexdigest())
        profile = score_profile(parent_record['result']) if parent_record else ()
        self.memory_query = parent, operator, parent_record, profile
        sections, ranking = retrieve(self.history, parent, operator, profile, self.memory_mode)
        if self.memory_gate and self.gate_choice is None:
            relevant = any(item['reason'] == 'relevance' and
                           (item['rank'][0] > 0 or item['rank'][1] > 0) for item in ranking)
            stagnant = not any(row['valid'] and row['gain'] > 0 for row in self.history[-8:])
            state = relevant, stagnant
            enabled = relevant
            if self.memory_gate == 'bandit' and ranking:
                if self.gate_rng.random() < .15:
                    enabled = bool(self.gate_rng.randrange(2))
                else:
                    enabled = max((False, True), key=lambda arm:
                                  (self.gate_values[state, arm][0] == 0,
                                   self.gate_values[state, arm][1]))
            self.gate_choice = state, enabled and bool(ranking), bool(ranking)
        if self.gate_choice:
            state, enabled, eligible = self.gate_choice
            event(self.folder, 'memory_gate_choice', state=state, enabled=enabled, eligible=eligible)
            if not enabled:
                sections = []
        if self.pending_feedback:
            expires, ids, evidence = self.pending_feedback
            if self.iteration > expires:
                self.pending_feedback = None
            elif parent.id in ids or parent.parent_id in ids:
                prompt = {**prompt, 'user': prompt['user'] +
                          '\n<<<AHD_OPTIONAL:addon_feedback>>>\nMeasured edit feedback (training only):\n'
                          + evidence + '\n<<<AHD_END_OPTIONAL>>>\n'}
                self.pending_feedback = None
                event(self.folder, 'addon_feedback_offered', iteration=self.iteration, parent=parent.id)
        if sections:
            memory = ''.join(f'\n<<<AHD_OPTIONAL:memory_{name}>>>\n{text}\n<<<AHD_END_OPTIONAL>>>\n'
                             for name, text in sections)
            prompt = {**prompt, 'user': prompt['user'] + memory}
        event(self.folder, 'memory_retrieval', parent_id=parent.id, operator=operator,
              mode=self.memory_mode, selected=ranking, sections=[name for name, _ in sections])
        return prompt

    async def _run_iteration(self, iteration, retry_times=1):
        self.iteration = iteration
        self.memory_query = None
        self.gate_choice = None
        observe = self.tuner and self.tuner.mode == 'correct'
        if observe:
            first_slot = json.loads((self.guard/'ledger.json').read_text())['used']
        if self.lever:
            database = self.database
            native = database.sample
            had_override = 'sample' in vars(database)
            self.lever.choice = None
            self.lever.duplicate = False
            started = time.monotonic()
            database.sample = lambda *args, **kwargs: self.lever.sample(
                self, native, iteration, *args, **kwargs)
            try:
                result = await super()._run_iteration(iteration, retry_times)
            finally:
                if had_override:
                    database.sample = native
                else:
                    del database.sample
            self.lever.observe(self, result, iteration, time.monotonic() - started)
        else:
            result = await super()._run_iteration(iteration, retry_times)
        if observe:
            used = json.loads((self.guard/'ledger.json').read_text())['used']
            rows = [json.loads(path.read_text()) for slot in range(first_slot, used)
                    if (path := self.guard/f'record_{slot:06d}.json').exists()]
            if any('Causality check failed' in (instance.get('error') or '')
                   for row in rows for batch in row['result'].get('instance_batches', [])
                   for instance in batch['results']):
                outcome = 'causality'
            elif rows:
                outcome = 'valid' if rows[-1]['valid'] else 'invalid evaluation'
            else:
                outcome = 'generation or parsing failure'
            event(self.folder, 'prompt_outcome', iteration=iteration, outcome=outcome)
            await self.tuner.observe(outcome)
        if self.memory_mode and self.memory_query:
            parent, operator, before, profile = self.memory_query
            self._index_scores()
            child = result.child_program_dict or {}
            source = child.get('solution')
            sha = hashlib.sha256(source.encode()).hexdigest() if source else None
            after = self.score_records.get(sha)
            observed = score_profile(after['result']) if after else ()
            valid = bool(after and after['valid'] and not result.error)
            if self.reuse and valid:
                code = self._active_search_algorithm_code
                if code:
                    key = hashlib.sha256(code.encode()).hexdigest()
                    if key not in self.reuse.archive:
                        self.reuse.archive[key] = code
                        event(self.folder, 'strategy_verified', iteration=iteration, strategy=key)
            issue = None
            if not valid:
                issue = ('timeout' if after and after['result'].get('n_timeout', 0) else
                         'invalid evaluated output' if after else 'generation or evaluation failed')
            row = dict(id=len(self.history)+1, iteration=iteration, parent_id=parent.id,
                       child_id=child.get('id'), parent_hash=before['hash'] if before else None,
                       child_hash=sha, operator=operator, valid=valid, issue=issue,
                       changed=changed_functions(parent.solution, source) if source else [],
                       gain=after['score']-before['score'] if after and before else 0.,
                       runtime=(after['result'].get('evaluation_seconds', 0.) if after else 0.),
                       case_deltas=[b-a for a,b in zip(profile, observed)] if len(profile)==len(observed) else [])
            if self.gate_choice and self.gate_choice[2]:
                state, enabled, _ = self.gate_choice
                reward = max(0., row['gain']) if valid else 0.
                if self.memory_gate == 'bandit':
                    count, mean = self.gate_values[state, enabled]
                    self.gate_values[state, enabled] = [count + 1, mean + (reward - mean) / (count + 1)]
                event(self.folder, 'memory_gate_outcome', iteration=iteration, state=state,
                      enabled=enabled, reward=reward)
            if self.addon in ('feedback', 'critique') and valid and before:
                evidence = measured_feedback(row, self.lever.incumbent, after['score'])
                if evidence:
                    event(self.folder, 'addon_feedback_triggered', iteration=iteration, evidence=evidence)
                    if self.addon == 'critique' and self.critique_calls < 4:
                        self.critique_calls += 1
                        delta = '\n'.join(unified_diff(parent.solution.splitlines(), source.splitlines(), n=1))
                        note = await critique(self.folder, self.config.llm.api_base, evidence, delta[:2000])
                        if note:
                            evidence += ' Suggested next edit: ' + note
                    self.pending_feedback = iteration + 4, {parent.id, child.get('id')}, evidence
            if self.memory_mode == 'delta' and source:
                delta = unified_diff(parent.solution.splitlines(), source.splitlines(), n=0)
                edited_lines = [line for line in list(delta)[2:]
                                if line.startswith(('+', '-')) and '<<<' not in line]
                row['code_delta'] = ' | '.join(line[:70] for line in edited_lines[:4])
            self.history.append(row)
            event(self.folder, 'memory_observation', **row)
        return result

    def _parse_llm_response(self, response, parent_solution, iteration, attempt, retry_times):
        if self.course['candidate_format'] == 'diff':
            parsed = super()._parse_llm_response(response, parent_solution, iteration, attempt, retry_times)
            if self.followup != 'diff' or parsed[0] is None:
                return parsed
            child, summary, _ = parsed
            try:
                if child.count(START) != 1 or child.count(END) != 1:
                    raise ValueError('Keep both EVOLVE markers')
                prefix, _, tail = child.partition(START)
                body, marker, suffix = tail.partition(END)
                expected_prefix, _, rest = self.initial.partition(START)
                expected_suffix = rest.partition(END)[2]
                if not marker or prefix != expected_prefix or suffix != expected_suffix:
                    raise ValueError('Keep the original scaffold outside the editable block')
                assemble_block(self.initial, body, self.course['task'])
            except (ValueError, SyntaxError) as exc:
                event(self.folder, 'candidate_rejected', iteration=iteration, attempt=attempt, error=str(exc))
                return None, None, str(exc)
            self.lever.duplicate = any(child == p.solution for p in self.database.programs.values())
            if ast.dump(ast.parse(child)) == ast.dump(ast.parse(parent_solution)):
                event(self.folder, 'candidate_rejected', iteration=iteration, attempt=attempt, error='AST unchanged')
                return None, None, 'The editable block did not change the parent program'
            return child, summary, None
        try:
            child = assemble_block(self.initial, response, self.course['task'])
        except (SyntaxError, ValueError) as exc:
            event(self.folder, 'candidate_rejected', iteration=iteration, attempt=attempt, error=str(exc))
            return None, None, str(exc)
        if self.lever:
            self.lever.duplicate = any(child == p.solution for p in self.database.programs.values())
        if self.followup == 'exact':
            self._index_scores()
            sha = hashlib.sha256(child.encode()).hexdigest()
            record = self.score_records.get(sha)
            if record and record['valid']:
                folder = self.folder/'rejected_exact'
                folder.mkdir(exist_ok=True)
                (folder/f'{iteration:06d}-{attempt:02d}-{sha}.py').write_text(child)
                event(self.folder, 'exact_repeat_rejected', iteration=iteration, attempt=attempt,
                      source_hash=sha, prior_evaluation=record['evaluation'])
                return None, None, 'This exact program was already evaluated; propose a different edit'
        if ast.dump(ast.parse(child)) == ast.dump(ast.parse(parent_solution)):
            event(self.folder, 'candidate_rejected', iteration=iteration, attempt=attempt, error='AST unchanged')
            return None, None, 'The editable block did not change the parent program'
        return child, 'Editable-block replacement', None

    def _observation(self):
        ledger = json.loads((self.guard/'ledger.json').read_text())
        best = max(0., float(self._get_best_score() or 0.))
        now = time.time()
        if best > self.last_score:
            self.last_improvement = now
        self.last_score = best
        left = max(0., ledger['deadline']-now)
        slots = max(0, self.course['evaluations']-ledger['used'])
        cap = self.course['seconds']
        recent = []
        for index in range(max(0, ledger['used']-8), ledger['used']):
            path = self.guard/f'record_{index:06d}.json'
            if path.exists():
                recent.append(json.loads(path.read_text()))
        profiles = [score_profile(r['result']) for r in recent if r['valid']]
        repeated = 1.-len(set(profiles))/len(profiles) if profiles else 0.
        features = np.array([1., left/cap, slots/self.course['evaluations'],
                             min(1., (now-self.last_improvement)/cap),
                             min(1., (now-self.strategy_since)/cap),
                             sum(r['valid'] for r in recent)/max(1, len(recent)),
                             repeated,
                             min(1., self.meta_seconds/max(1., left))])
        allowed = self._meta_llm_available and slots >= 2 and left > max(60., 2*self.meta_seconds)
        return features, best, ledger['used'], allowed

    def _finish_action(self, features, best, used, allowed, terminal=False):
        if self.previous is None:
            return
        option, old_features, old_best, old_used, started, old_tick = self.previous
        action, commitment = self.policy.options[option]

        reward = best/(1.+best) - old_best/(1.+old_best)
        if self.reuse:
            self.reuse.update(reward)
        if self.pacing and not terminal:
            self.pacing.observe(action, time.time()-started, used-old_used, reward)
        target, prediction = self.policy.update(option, old_features, reward, features, allowed, terminal)
        event(self.folder, 'meta_transition', action=action, commitment=commitment,
              observed_callbacks=self.boundary_ticks-old_tick, reward=reward, target=target,
              prediction=prediction, slots=used-old_used, seconds=time.time()-started,
              terminal=terminal, features=old_features.tolist(), next_features=features.tolist())
        self.previous = None

    def _should_evolve_search(self):
        if self.policy is None:
            return super()._should_evolve_search()
        if self.policy.kind == 'rule':
            self.rule_due = super()._should_evolve_search() or self.rule_due
        self.boundary_ticks += 1
        if self.boundary_ticks < self.next_boundary:
            if not self.interruption or self.previous is None or self.boundary_ticks != self.previous[-1]+2:
                return False
            features, best, used, _ = self._observation()
            _, _, previous_best, previous_used, _, _ = self.previous
            valid = any(
                (path := self.guard/f'record_{slot:06d}.json').exists() and
                json.loads(path.read_text())['valid'] for slot in range(previous_used, used))
            state = (best > previous_best, valid, features[1] < .5)
            stop = self.interruption.decide(state, best/(1.+best))
            event(self.folder, 'option_interruption', state=[bool(value) for value in state],
                  stop=stop, evaluations=used, callbacks=self.boundary_ticks)
            if not stop:
                return False
        features, best, used, allowed = self._observation()
        self._finish_action(features, best, used, allowed)
        if allowed and not self.initialization_attempted and not self.search_controller.database.programs:
            self.initialization_attempted = True
            self.rule_due = False
            self.next_boundary = self.boundary_ticks + 4
            event(self.folder, 'meta_initialization', evaluations=used)
            return True
        penalty = 0.
        if self.pacing:
            paced, penalty, slack = self.pacing.decision(
                allowed, features[1]*self.course['seconds'],
                self.course['evaluations']-used, self.meta_seconds)
            event(self.folder, 'addon_pacing', eligible=bool(allowed), paced=bool(paced),
                  penalty=penalty, slack_seconds=slack, costs=self.pacing.costs)
            allowed = paced
        option = self.policy.choose(features, allowed, self.rule_due, penalty)
        action, commitment = self.policy.options[option]
        self.next_boundary = self.boundary_ticks + commitment
        self.rule_due = False if action else self.rule_due
        self.previous = option, features, best, used, time.time(), self.boundary_ticks
        event(self.folder, 'meta_decision', action=action, commitment=commitment, meta_available=bool(allowed),
              features=features.tolist(), evaluations=used, best_train=best)
        return bool(action)

    async def _evolve_search(self, solution_iter):
        start = time.monotonic()
        previous = self._active_search_algorithm_code
        if self.reuse and self.search_controller.database.programs:
            current = hashlib.sha256(previous.encode()).hexdigest() if previous else None
            remaining = self._observation()[0][1] * self.course['seconds']
            key, eligible = self.reuse.choose(current, remaining)
            if key and self._pending_search_result:
                await self._finalize_pending_search()
            switched = bool(key and self._switch_to_new_search_algorithm(SerializableResult(
                child_program_dict={'id': key, 'solution': self.reuse.archive[key]},
                iteration=solution_iter)))
            event(self.folder, 'strategy_reuse', iteration=solution_iter, eligible=eligible,
                  proposed=key, switched=switched, seconds=time.monotonic()-start)
            if switched:
                self.strategy_since = time.time()
                self._reset_search_window(start_iteration=solution_iter)
                return
            if key:
                del self.reuse.archive[key]
        try:
            await super()._evolve_search(solution_iter)
        finally:
            elapsed = time.monotonic()-start
            self.meta_seconds = .8*self.meta_seconds + .2*elapsed
            active = self._active_search_algorithm_code
            if active != previous:
                self.strategy_since = time.time()
            event(self.folder, 'meta_call', iteration=solution_iter, seconds=elapsed,
                  strategy_changed=active != previous, failures=self._meta_evolution_failures,
                  strategy_sha256=hashlib.sha256((active or '').encode()).hexdigest())

    async def run_discovery(self, *args, **kwargs):
        if self.tuner:
            await self.tuner.revise('prompt_draft')
        result = await super().run_discovery(*args, **kwargs)
        if self.interruption:
            _, best, _, _ = self._observation()
            self.interruption.update(best/(1.+best))
        if self.policy is not None:
            self._finish_action(*self._observation(), terminal=True)
        return result


if __name__ == '__main__':
    register_controller('evox', ExperimentalEvoX)
    raise SystemExit(main())
