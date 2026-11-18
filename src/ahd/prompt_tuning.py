"""Budgeted prompt drafting and correction for the EvoX block interface."""
import asyncio
from collections import Counter, deque
import json
import time
import urllib.request

from .audit import event

PROMPT_MODES = {'evox_prompt_draft': 'draft', 'evox_prompt_correct': 'correct'}


class PromptTuner:
    def __init__(self, folder, endpoint, mode, specification, starter):
        self.folder = folder
        self.endpoint = endpoint.rstrip('/') + '/chat/completions'
        self.mode = mode
        self.specification = specification
        self.starter = starter
        self.instruction = ''
        self.recent = deque(maxlen=8)
        self.corrections = 0

    async def revise(self, purpose, trigger=None):
        ledger = json.loads((self.folder/'guard/ledger.json').read_text())
        remaining = ledger['deadline'] - time.time()
        if remaining < 120:
            event(self.folder, 'prompt_edit_skipped', purpose=purpose, reason='less than 120 seconds remaining')
            return
        system = ('Write one concise natural-language mutation instruction for a code generator. '
                  'Return one plain-text paragraph under 500 characters. Preserve the task rules, '
                  'evaluation constraints and editable-block format. Include no code or new examples.')
        user = f'Task specification:\n{self.specification}\n\nStarter program:\n{self.starter}'
        if trigger:
            user += (f'\n\nCurrent instruction: {self.instruction}\nRecurring observed failure: {trigger}. '
                     f'Outcomes since last edit: {dict(Counter(self.recent))}. Revise the instruction to address it.')
        payload = json.dumps(dict(ahd_purpose=purpose, messages=[
            dict(role='system', content=system), dict(role='user', content=user)])).encode()

        def request():
            req = urllib.request.Request(self.endpoint, data=payload,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=min(60., remaining-60.)) as response:
                return json.load(response)['choices'][0]

        try:
            choice = await asyncio.to_thread(request)
            instruction = choice['message']['content'].strip()
            if (choice['finish_reason'] != 'stop' or not 0 < len(instruction) <= 500
                    or '\n' in instruction or '`' in instruction or '<<<' in instruction):
                raise ValueError('Expected one plain-text paragraph under 500 characters')
        except (OSError, ValueError, KeyError, IndexError, AttributeError) as exc:
            event(self.folder, 'prompt_edit_failed', purpose=purpose, trigger=trigger, error=str(exc))
            return
        previous = self.instruction
        self.instruction = instruction
        event(self.folder, 'prompt_edit', purpose=purpose, trigger=trigger, before=previous, after=instruction)

    async def observe(self, outcome):
        if self.mode != 'correct' or self.corrections >= 2:
            return
        self.recent.append(outcome)
        if outcome != 'valid' and self.recent.count(outcome) >= 3:
            self.corrections += 1
            await self.revise('prompt_correction', outcome)
            self.recent.clear()
