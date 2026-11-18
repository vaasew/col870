"""Imported by upstream SkyDiscover. Hard cap covers initial and retry evaluations."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
from ahd.evaluation import evaluate as official_evaluate
from ahd.protocol import atomic_json, digest
from ahd.audit import event, failure


def upstream_result(metrics, feedback):
    from skydiscover.optimize.evaluation.evaluation_result import EvaluationResult
    return EvaluationResult(metrics=metrics, artifacts={'feedback': feedback})


def evaluate(program_path):
    root = Path(os.environ['AHD_GUARD_DIR'])
    config = json.loads((root.parent/'manifest.json').read_text())['config']
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger_path = root/'ledger.json'
        ledger = json.loads(ledger_path.read_text())
        if ledger['used'] >= config['evaluations'] or time.time() >= ledger['deadline']-5:
            return upstream_result({'combined_score': 0.}, 'Run budget exhausted')
        number = ledger['used']
        ledger['used'] += 1
        atomic_json(ledger_path, ledger)
        path = root/f'program_{number:06d}.py'
        shutil.copyfile(program_path, path)
        source_hash = digest(path)
        atomic_json(root/f'reservation_{number:06d}.json',
                    dict(number=number, path=str(path.relative_to(root.parent)),
                         hash=source_hash, started=time.time()))
    event(root.parent, 'evaluation_reserved', number=number+1, path=str(path.relative_to(root.parent)))
    try:
        result = official_evaluate(config['task'], path, timeout=max(.01, ledger['deadline']-time.time()-5),
                                  parallel=config['parallel'], work=root/f'eval_{number:06d}.log')
        if digest(path) != source_hash:
            raise RuntimeError('Candidate source changed during evaluation')
    except Exception as exc:
        failure(root.parent, exc, evaluation=number+1)
        raise
    record = dict(id=number, path=str(path.relative_to(root.parent)), score=result['combined_score'],
                  hash=digest(path), feedback=result.get('feedback', ''), result=result,
                  valid=result.get('n_failed', 0)==0, parent=None if number==0 else 'upstream',
                  evaluation=number+1, elapsed=config['seconds']-(ledger['deadline']-time.time()))
    atomic_json(root/f'record_{number:06d}.json', record)
    if not result.get('outer_timeout'):
        with (root/'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            best_path = root/'best.json'
            best = json.loads(best_path.read_text()) if best_path.exists() else None
            if best is None or record['score'] > best['score']:
                temp = root.parent/'best_train.py.tmp'
                shutil.copyfile(path, temp)
                temp.replace(root.parent/'best_train.py')
                atomic_json(best_path, record)
    event(root.parent, 'evaluation_completed', evaluation=number+1, score=record['score'],
          elapsed=record['elapsed'], valid=record['valid'])
    return upstream_result({k:v for k,v in result.items() if isinstance(v,(float,int))
                            and k != 'completed_at'},
                           result.get('feedback',''))
