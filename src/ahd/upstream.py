"""Actual pinned upstream algorithms; local proxy enforces generator settings.
Budget guard is an adapter, not a reimplementation of EvoX.
"""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import yaml
from .generator import LiveGenerator
from .audit import event, failure
from .evaluation import terminate
from .memory import MEMORY_BUDGET, MEMORY_MODES
from .causality import CAUSALITY_POLICY
from .addons import ADDONS
from .followups import FOLLOWUPS
from .search_levers import SEARCH_LEVERS
from .prompt_tuning import PROMPT_MODES
from .protocol import ROOT, DECODING, INSTANCE_SECONDS, atomic_json, task_path, validate_config, wall_limit

VARIANTS = {
    'evox_lineage_memory': ('block', 'bandit'),
    'evox_lineage_reward': ('block', 'bandit'),
}
DURATION_VARIANTS = set()
MEMORY_GATES = {}
TARGETED_HINTS = {}
SIGNAL_ARMS = {}
SIGNAL_SPRINT_HINTS = {}

OPTIONAL_CONTEXT = ('addon_feedback', 'mutation_guidance', 'other_context_programs', 'previous_attempts', 'search_guidance')
OPTIONAL_BLOCK = re.compile(r'(?ms)^<<<AHD_OPTIONAL:(\w+)>>>\n(.*?)^<<<AHD_END_OPTIONAL>>>\n')


def separate_memory(messages):
    plain, sections = [], []
    for message in messages:
        content = message.get('content')
        if message.get('role') != 'user' or not isinstance(content, str):
            plain.append(message)
            continue
        marker = content.find('\n<<<AHD_OPTIONAL:memory_')
        if marker == -1:
            plain.append(message)
            continue
        sections.extend((m[1].removeprefix('memory_'), m[2].strip())
                        for m in OPTIONAL_BLOCK.finditer(content[marker:]) if m[1].startswith('memory_'))
        plain.append({**message, 'content': content[:marker]})
    return plain, sections


def prompt_variants(messages):
    """Try full context, then drop optional sections without cutting parent code."""
    omitted = set()
    for section in (None, *OPTIONAL_CONTEXT):
        if section:
            omitted.add(section)
        yield [
            {**message, 'content': OPTIONAL_BLOCK.sub(
                lambda match: '' if match[1] in omitted else match[2], message['content'])}
            if message.get('role') == 'user' and isinstance(message.get('content'), str) else message
            for message in messages
        ]



def launch(a):
    upstream = Path(a.upstream).resolve()
    client = LiveGenerator(a.api_base, a.model, a.server_manifest)
    started_at = time.time()
    variant = getattr(a, 'variant', 'evox')
    candidate_format, scheduler = VARIANTS[variant]
    targeted = variant == 'evox_targeted'
    if targeted and a.task not in TARGETED_HINTS:
        raise ValueError('Targeted pilot supports Cloudcast and Signal Processing')
    signal_variant = variant in SIGNAL_ARMS or variant in SIGNAL_SPRINT_HINTS
    if signal_variant and a.task != 'signal_processing':
        raise ValueError('Signal pilot variants require signal_processing')
    guided_ar, specialists = SIGNAL_ARMS.get(variant, (False, False))
    config_variant = 'evox_lineage_reward' if targeted or signal_variant else variant
    frozen = variant in ADDONS or variant in FOLLOWUPS
    c = dict(task=a.task, policy='evox', seed=a.seed, evaluations=a.evaluations,
             seconds=a.seconds if a.seconds is not None else wall_limit(a.task), parallel=a.parallel, gamma=1., test_only=False,
             prompt_profile=(variant if variant in SIGNAL_SPRINT_HINTS else
                             'gated_ar' if guided_ar else 'targeted' if targeted else 'starter'),
             specialist_fraction=.25 if specialists else 0., lineage_lambda=(1.0 if variant == 'evox_lineage_memory' else 0.25), variant=variant, stage=getattr(a, 'stage', 'custom'),
             candidate_format=candidate_format, scheduler=scheduler,
             commitment_lengths=[4, 2, 8] if variant in DURATION_VARIANTS else [4],
             causality_policy=CAUSALITY_POLICY if a.task == 'signal_processing' else None,
             addon=ADDONS.get(variant), followup=FOLLOWUPS.get(variant),
             search_lever=SEARCH_LEVERS.get('evox_lineage_memory' if frozen else config_variant),
             memory_gate=MEMORY_GATES.get(variant),
             memory_mode=MEMORY_MODES.get('evox_lineage_memory' if frozen else config_variant),
             prompt_mode=PROMPT_MODES.get(variant),
             memory_budget=MEMORY_BUDGET if config_variant in MEMORY_MODES or frozen else 0)
    validate_config(c)
    deadline = min(started_at + c['seconds'], a.deadline or float('inf'))
    if deadline - time.time() <= 15:
        raise TimeoutError('Insufficient task budget after server startup')
    folder = Path(a.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    guard = folder/'guard'
    guard.mkdir()
    atomic_json(folder/'manifest.json', dict(config=c, server=client.manifest, decoding=DECODING,
                task_start=deadline-c['seconds'], adapter_start=started_at, deadline=deadline))
    event(folder, 'run_start', deadline=deadline)
    atomic_json(guard/'ledger.json', dict(used=0, deadline=deadline))
    counter = [0]
    upstream_calls = 0
    prompt_calls = 0
    mutex = threading.Lock()

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.respond(200, dict(object='list', data=[dict(id=a.model, object='model')]))

        def do_POST(self):
            nonlocal upstream_calls, prompt_calls
            call_start = time.monotonic()
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            purpose = request.get('ahd_purpose', 'upstream')
            with mutex:
                index = counter[0]
                counter[0] += 1
                if purpose in ('prompt_draft', 'prompt_correction', 'addon_critique'):
                    call_seed = a.seed + 1000000 + prompt_calls
                    prompt_calls += 1
                else:
                    call_seed = a.seed + upstream_calls
                    upstream_calls += 1
            loc = folder/f'llm_{index:06d}'
            loc.mkdir()
            atomic_json(loc/'upstream_request.json', request)
            atomic_json(loc/'enforced_settings.json', dict(model=a.model, seed=call_seed, **DECODING))
            try:
                if time.time() >= deadline-5:
                    raise TimeoutError('Run deadline exhausted')
                call_client = LiveGenerator(a.api_base, a.model, a.server_manifest)
                call_client.audit_dir = loc/'http'
                messages, memory = separate_memory(request['messages']) if c['memory_mode'] else (request['messages'], [])
                variants = list(prompt_variants(messages))
                distinct = [variant for i, variant in enumerate(variants) if variant not in variants[:i]]
                body = call_client.generate(distinct[0], call_seed, deadline-time.time()-5,
                                            fallbacks=distinct[1:], memory=memory, memory_budget=c['memory_budget'])
                atomic_json(loc/'response.json', body)
                event(folder, 'llm_completed', call=index, usage=body.get('usage'),
                      purpose=purpose,
                      seconds=time.monotonic()-call_start,
                      feedback_sent=call_client.feedback_sent,
                      memory_offered=[name for name, _ in memory],
                      memory_sent=call_client.memory_sent, memory_tokens=call_client.memory_tokens,
                      finish_reason=body['choices'][0].get('finish_reason'))
                self.respond(200, body)
            except Exception as exc:
                atomic_json(loc/'error.json', {'error': str(exc)})
                event(folder, 'llm_failed', call=index, purpose=purpose,
                      seconds=time.monotonic()-call_start, error=str(exc))
                self.respond(400, {'error': {'message': str(exc), 'type': type(exc).__name__}})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    template = yaml.safe_load((task_path(a.task)/'config.yaml').read_text())
    template.update(dict(max_iterations=max(0, a.evaluations-1), checkpoint_interval=1,
                         max_parallel_iterations=1))
    task_prompt = (task_path(a.task)/'prompt.txt').read_text()
    if targeted:
        task_prompt += '\n\nSearch direction:\n' + TARGETED_HINTS[a.task]
    if guided_ar:
        task_prompt += (
            '\n\nSearch direction: Try an adaptive autoregressive predictor that uses '
            'recent samples. Use a causal oscillation detector requiring sustained evidence '
            'before switching toward AR. Otherwise use EMA with hysteresis to preserve steps. '
            'Keep outputs aligned and prefix-invariant.'
        )
    if variant in SIGNAL_SPRINT_HINTS:
        task_prompt += '\n\nSearch direction: ' + SIGNAL_SPRINT_HINTS[variant]
    template['prompt'] = dict(system_message=task_prompt)


    prompt_templates = folder/'prompt_templates'
    prompt_templates.mkdir()
    candidate_templates = prompt_templates
    if candidate_format == 'block' or variant == 'evox_lineage_diff':
        candidate_templates = folder/'candidate_templates'
        candidate_templates.mkdir()
        template['diff_based_generation'] = candidate_format == 'diff'
    builders = upstream/'skydiscover/optimize/context_builder'
    for directory in (builders/'default/templates',):
        for path in directory.glob('*.txt'):
            text = path.read_text()
            original = text
            if '{timeout_warning}' in text:
                text = text.replace('{timeout_warning}',
                    f'- Entry-point runtime limit: {INSTANCE_SECONDS[a.task]} seconds per instance. '
                    'This is distinct from the outer discovery budget.')
            for name in OPTIONAL_CONTEXT:
                placeholder = '{' + name + '}'
                text = text.replace(placeholder, f'<<<AHD_OPTIONAL:{name}>>>\n{placeholder}\n<<<AHD_END_OPTIONAL>>>\n')
            if text != original:
                (prompt_templates/path.name).write_text(text)
            if candidate_format == 'block':
                if path.name == 'full_rewrite_user_message.txt':
                    text = text.replace('Provide the complete new program solution.',
                        'Return only the full replacement for the code between the EVOLVE markers. '
                        'The original outer scaffold will be restored automatically.')
                    if a.task == 'signal_processing':
                        text += ('\nThe fixed run_signal_processing wrapper calls process_signal. '
                                 'Define process_signal once and make it call your improved filter. '
                                 'Leave run_signal_processing out of the editable block.\n')
                (candidate_templates/path.name).write_text(text)
            if variant == 'evox_lineage_diff':
                if 'diff' in path.name:
                    text += '\nKeep both EVOLVE markers and all code outside them unchanged.\n'
                    if a.task == 'signal_processing':
                        text += ('The fixed run_signal_processing wrapper calls process_signal. '
                                 'Define process_signal once and leave the wrapper unchanged.\n')
                (candidate_templates/path.name).write_text(text)
    template['prompt']['template_dir'] = str(candidate_templates)
    template['llm'] = dict(api_base=f'http://127.0.0.1:{server.server_port}/v1',
                           models=[dict(name=a.model)], temperature=.99, top_p=.95,
                           max_tokens=8000, retries=0, timeout=600)
    template['search'] = dict(type='evox', share_llm=True, database=dict(random_seed=a.seed, auto_generate_variation_operators=True))
    template['evaluator'] = dict(timeout=int(c['seconds']), max_retries=0,
                                 cascade_evaluation=False, inject_evaluator_context=False)

    meta_dir = upstream/'skydiscover/optimize/search/evox/config'
    meta = yaml.safe_load((meta_dir/'search.yaml').read_text())
    meta['prompt']['system_message'] = (meta_dir/'evox_search_sys_prompt.txt').read_text()
    meta['prompt']['template_dir'] = str(prompt_templates)
    meta['llm'] = dict(template['llm'])
    meta['search']['database'] = dict(random_seed=a.seed)
    meta_path = folder/'evox-meta.yaml'
    meta_path.write_text(yaml.safe_dump(meta))
    template['search']['database']['config_path'] = str(meta_path)

    cfg = folder/'upstream.yaml'
    cfg.write_text(yaml.safe_dump(template))
    env = os.environ.copy()
    env['AHD_GUARD_DIR'] = str(guard)
    env['OPENAI_API_KEY'] = 'local-proxy'
    env['PYTHONPATH'] = str(ROOT)+os.pathsep+str(upstream)+os.pathsep+env.get('PYTHONPATH', '')
    entry = 'skydiscover' if variant == 'evox' else 'ahd.evox_variant'
    cmd = [sys.executable, '-m', entry, 'optimize',
           str(task_path(a.task)/'initial_program.py'), str(ROOT/'ahd/guarded_evaluator.py'),
           '--config', str(cfg), '--output', str(folder/'upstream')]
    atomic_json(folder/'command.json', cmd)
    def cancel(*_):
        raise KeyboardInterrupt('EvoX job interrupted')

    previous_handler = signal.signal(signal.SIGTERM, cancel)
    try:
        with (folder/'upstream.log').open('w') as log:
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
            hit_deadline = False
            try:
                while p.poll() is None:
                    left = deadline-time.time()-5
                    if left <= 0:
                        hit_deadline = True
                        terminate(p)
                        break
                    best_path = guard/'best.json'
                    best = json.loads(best_path.read_text())['score'] if best_path.exists() else None
                    atomic_json(folder/'heartbeat.json', dict(time=time.time(), deadline=deadline,
                                llm_calls=counter[0], best_train=best,
                                evaluations=json.loads((guard/'ledger.json').read_text())['used']))
                    try:
                        p.wait(timeout=min(30, left))
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException:
                terminate(p)
                raise
        finished_at = time.time()
        used = json.loads((guard/'ledger.json').read_text())['used']
        stop_reason = ('wall_clock' if hit_deadline else 'upstream_failure' if p.returncode else
                       'evaluation_cap' if used >= c['evaluations'] else 'upstream_finished')
        atomic_json(folder/'exit.json', dict(returncode=p.returncode, llm_calls=counter[0],
                    elapsed=finished_at-started_at, finished_at=finished_at,
                    task_wall_seconds=finished_at-(deadline-c['seconds']),
                    evaluations=used, stop_reason=stop_reason))
        if p.returncode and not hit_deadline:
            raise RuntimeError(f'Upstream failed; inspect {folder}/upstream.log')
        event(folder, 'run_stop', returncode=p.returncode, evaluations=used, stop_reason=stop_reason)
    except BaseException as exc:
        failure(folder, exc)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        server.shutdown()
        server.server_close()
