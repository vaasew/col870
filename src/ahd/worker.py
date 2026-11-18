"""Fresh evaluator process: isolates task module names and cleans children on TERM.
Process isolation enforces time limits; it is not a security sandbox for hostile code.
"""
import importlib.util
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from .protocol import atomic_json


def main():
    task_dir, program, split, output, parallel = sys.argv[1:]
    os.environ['EVAL_PARALLEL'] = parallel
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    def stop(*_):
        raise KeyboardInterrupt('evaluation cancelled')
    signal.signal(signal.SIGTERM, stop)

    if sys.platform.startswith('linux'):
        import ctypes
        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent:
            raise KeyboardInterrupt('parent exited during worker initialization')
    try:
        spec = importlib.util.spec_from_file_location('task_evaluator', Path(task_dir) / 'evaluator.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        signal_task = Path(task_dir).name == 'signal_processing'
        if signal_task:
            from .causality import CAUSALITY_POLICY, install
            install(mod)

        instance_batches = []
        summarize = mod.common.summarize
        def capture(results, labels=None, **extra):
            instance_batches.append(dict(results=results, labels=labels, extra=extra))
            return summarize(results, labels, **extra)
        mod.common.summarize = capture
        result = mod.evaluate(program, split=split)
        if not isinstance(result, dict):
            result = {**result.metrics, **result.artifacts}
        result['instance_batches'] = instance_batches
        if signal_task:
            result['causality_policy'] = CAUSALITY_POLICY
            result['causality_passed'] = result.get('n_failed') == 0
            if any('Causality check failed' in (row.get('error') or '')
                   for batch in instance_batches for row in batch['results']):
                result['combined_score'] = 0.

        if split == 'train' and instance_batches:
            entries=[]
            for batch in instance_batches:
                labels=batch['labels'] or [str(i) for i in range(len(batch['results']))]
                entries.extend((lab,r) for lab,r in zip(labels,batch['results']))
            worst=sorted(entries,key=lambda item:item[1]['score'])[:5]
            text='Lowest-scoring training instances (diagnostic; optimize the full mean):\n'
            for lab, r in worst:
                text += f"{lab}: score={r['score']:.9g}, status={r['status']}, runtime={r['program_time']}"
                if r.get('diagnostics'):
                    text += ', ' + ', '.join(f'{key}={value:.5g}' for key, value in r['diagnostics'].items())
                if signal_task and r.get('error'):
                    text += ', error=' + r['error'][:350]
                text += '\n'
            result['feedback']=text+'\n\n'+result.get('feedback','')
        result['completed_at'] = time.time()
        atomic_json(output, result)
    except Exception:
        atomic_json(output, {'infrastructure_error': traceback.format_exc()})
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
