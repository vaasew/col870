"""Optional pacing and evaluator feedback for the frozen lineage method."""
import asyncio
import json
import time
import urllib.request

from .audit import event

ADDONS = {
    'evox_add_pacing': 'pacing',
    'evox_add_cost': 'cost',
    'evox_add_feedback': 'feedback',
    'evox_add_critique': 'critique',
}


class Pacing:
    def __init__(self, mode):
        self.mode = mode
        self.costs = [[0, 0., 0.], [0, 0., 0.]]
        self.positive_gains = []

    def observe(self, action, seconds, slots, reward):
        row = self.costs[action]
        row[0] += 1
        for index, observed in enumerate((seconds, slots), 1):
            row[index] += (observed - row[index]) / row[0]
        if reward > 0:
            self.positive_gains.append(reward)

    def decision(self, allowed, seconds, slots, meta_seconds):
        count, duration, used = self.costs[0]
        if not allowed or count < 2 or used <= 0:
            return allowed, 0., None
        slack = seconds - slots * duration / used

        paced = meta_seconds <= max(0., slack) + duration
        penalty = 0.
        if self.mode == 'cost' and min(row[0] for row in self.costs) >= 2 and self.positive_gains:
            _, meta_duration, meta_slots = self.costs[1]
            scale = sum(self.positive_gains) / len(self.positive_gains)
            penalty = scale * (max(0., meta_duration-duration) / max(1., seconds)
                               + max(0., meta_slots-used) / max(1, slots))
        return paced, penalty, slack


def measured_feedback(row, incumbent, score):
    if not row['valid']:
        return None
    deltas = row['case_deltas']
    tradeoff = bool(deltas and min(deltas) < -1e-9 and max(deltas) > 1e-9)
    below_best = incumbent is not None and row['gain'] > 1e-9 and score < incumbent - 1e-9
    if not (tradeoff or below_best):
        return None
    changed = ', '.join(row['changed'][:4])[:140] or 'module-level edit'
    evidence = f'Edit {row["id"]}, functions {changed}. Child minus parent: {row["gain"]:+.5g}.'
    if tradeoff:
        low = min(range(len(deltas)), key=deltas.__getitem__)
        high = max(range(len(deltas)), key=deltas.__getitem__)
        evidence += f' Largest gain: case {high+1} ({deltas[high]:+.4g}); regression: case {low+1} ({deltas[low]:+.4g}).'
    if below_best:
        evidence += f' Child minus previous incumbent: {score-incumbent:+.5g}.'
    return evidence


async def critique(folder, endpoint, evidence, code_delta):
    remaining = json.loads((folder/'guard/ledger.json').read_text())['deadline'] - time.time()
    if remaining < 120:
        event(folder, 'addon_critique_skipped', reason='insufficient time')
        return ''
    payload = json.dumps(dict(ahd_purpose='addon_critique', messages=[
        dict(role='system', content='Suggest one next edit in at most two short sentences. '
             'Use the measured training evidence and code delta. Separate hypotheses from observations. '
             'Return plain text only, no code.'),
        dict(role='user', content=f'{evidence}\nCode delta excerpt:\n{code_delta}')])).encode()

    def request():
        req = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
                                     data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=min(60., remaining-60.)) as response:
            return json.load(response)['choices'][0]

    try:
        choice = await asyncio.to_thread(request)
        note = ' '.join(choice['message']['content'].split())
        if choice['finish_reason'] != 'stop' or not 0 < len(note) <= 500 or '`' in note or '<<<' in note:
            raise ValueError('Invalid critique response')
    except (OSError, ValueError, KeyError, IndexError, AttributeError, TypeError) as exc:
        event(folder, 'addon_critique_failed', error=str(exc))
        return ''
    return note
