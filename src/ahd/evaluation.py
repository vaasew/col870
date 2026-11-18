import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from .protocol import ROOT, task_path, atomic_json


def terminate(p):
    if p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def evaluate(task, program, split='train', timeout=600, parallel=4, work=None):
    if split not in ('train', 'val'):
        raise ValueError('Invalid split')
    with tempfile.TemporaryDirectory(prefix='ahd-eval-') as temp:
        out = Path(work).with_suffix('.result.json') if work else Path(temp) / 'result.json'
        out.unlink(missing_ok=True)
        log_path = Path(work) if work else Path(temp) / 'worker.log'
        start = time.monotonic()
        with log_path.open('w') as log:
            p = subprocess.Popen([sys.executable, '-m', 'ahd.worker', str(task_path(task)),
                                  str(Path(program).resolve()), split, str(out), str(parallel)],
                                 cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
            try:
                p.wait(timeout=max(0.01, timeout))
            except subprocess.TimeoutExpired:
                terminate(p)
                result = dict(combined_score=0.0, n_failed=1, n_instances=1,
                            feedback='Outer evaluation deadline exhausted', outer_timeout=True,
                            evaluation_seconds=time.monotonic() - start)
                if work:
                    atomic_json(Path(work).with_suffix('.result.json'), result)
                return result
            except BaseException:
                terminate(p)
                raise
        if not out.exists():
            raise RuntimeError(f'Evaluator exited {p.returncode} without output: {log_path}')
        result = json.loads(out.read_text())
        result['evaluation_seconds'] = time.monotonic() - start
        if work:
            atomic_json(Path(work).with_suffix('.result.json'), result)
        if 'infrastructure_error' in result:
            raise RuntimeError(result['infrastructure_error'])
        if result.get('n_instances', 0) <= 0:
            raise RuntimeError('Evaluator returned no instances')
        score = float(result['combined_score'])
        if not math.isfinite(score) or score < 0:
            raise RuntimeError(f'Invalid evaluator score: {score}')
        result['evaluation_seconds'] = time.monotonic() - start
        return result
