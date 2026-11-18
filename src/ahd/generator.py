"""vLLM-compatible client, exact server-side token counting, no automatic retries."""
import json
import os
from pathlib import Path
import time
import urllib.request
from .protocol import DECODING, MODEL, atomic_json


class ContextOverflow(ValueError):
    pass


class LiveGenerator:
    def __init__(self, base, model, manifest):
        self.audit_dir = None
        self._request_count = 0
        self.base = base.rstrip('/')
        self.model = model
        self.manifest = json.loads(Path(manifest).read_text())
        m = self.manifest
        if m.get('model') != MODEL or m.get('dtype') != 'bfloat16' or not m.get('revision') or m['revision'].startswith('REPLACE'):
            raise ValueError('Server manifest must attest exact model, BF16, and nonempty revision')
        if m.get('quantization') is not None or m.get('max_model_len') != 16384:
            raise ValueError('No quantization; context must be 16384')
        if m.get('served_model_name') != model:
            raise ValueError('Served model alias does not match manifest')


    def request(self, url, payload, timeout):
        headers = {'Content-Type': 'application/json'}
        key = os.getenv('AHD_API_KEY')
        if key:
            headers['Authorization'] = f'Bearer {key}'
        request = urllib.request.Request(url, json.dumps(payload).encode(), headers)
        audit_path = None
        if self.audit_dir:
            root = Path(self.audit_dir)
            root.mkdir(parents=True, exist_ok=True)
            audit_path = root/f'http_{self._request_count:06d}.json'
            self._request_count += 1
            atomic_json(audit_path, dict(url=url, request=payload, status='started', time=time.time()))
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=max(0.01, timeout)) as r:
                result = json.load(r)
            if audit_path:
                atomic_json(audit_path, dict(url=url, request=payload, response=result, status='ok',
                                            seconds=time.monotonic()-start))
            return result
        except Exception as exc:
            detail = exc.read(65536).decode(errors='replace') if hasattr(exc, 'read') else None
            if audit_path:
                atomic_json(audit_path, dict(url=url, request=payload, status='error', error=str(exc),
                                            response_body=detail, seconds=time.monotonic()-start))
            raise

    def generate(self, messages, seed, remaining, *, fallbacks=(), memory=(), memory_budget=0):
        start = time.monotonic()
        limit = 16384 - DECODING['max_tokens']
        self.memory_sent = []
        self.memory_tokens = 0

        def tokenize(prompt):
            left = remaining - (time.monotonic() - start)
            if left <= 0:
                raise TimeoutError('Deadline exhausted while tokenizing')
            return self.request(self.base.removesuffix('/v1') + '/tokenize',
                                dict(model=self.model, messages=prompt, add_generation_prompt=True),
                                min(30, left))['count']

        for messages in (messages, *fallbacks):
            count = tokenize(messages)
            if count <= limit:
                break
        else:
            raise ContextOverflow(f'Prompt has {count} tokens; only {limit} allowed')
        if memory:
            user_index = max(i for i, m in enumerate(messages) if m['role'] == 'user')
            for size in range(len(memory), 0, -1):
                evidence = '\n'.join(section for _, section in memory[:size])
                enriched = list(messages)
                user = messages[user_index]
                enriched[user_index] = {**user, 'content': user['content'] +
                                        '\n\nPrior search evidence (training only):\n' + evidence}
                total = tokenize(enriched)
                if total <= limit and total-count <= memory_budget:
                    self.memory_sent = [name for name, _ in memory[:size]]
                    self.memory_tokens = max(0, total-count)
                    messages = enriched
                    break
        self.feedback_sent = any(
            message.get('role') == 'user' and 'Measured edit feedback (training only):' in message['content']
            for message in messages)
        left = remaining - (time.monotonic() - start)
        if left <= 0:
            raise TimeoutError('Deadline exhausted while tokenizing')
        body = self.request(self.base + '/chat/completions',
                            dict(model=self.model, messages=messages, seed=seed, **DECODING),
                            min(600, left))
        return body

