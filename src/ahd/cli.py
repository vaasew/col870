"""Entry points for the pinned EvoX course adapter."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import time

from .evaluation import evaluate
from .protocol import TASKS, atomic_json, digest
from .upstream import VARIANTS, launch
from .causality import CAUSALITY_POLICY


def records(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text())
    recs = [json.loads(p.read_text()) for p in sorted((folder/'guard').glob('record_*.json'))]
    known = {r['id'] for r in recs}


    for path in sorted((folder/'guard').glob('eval_*.result.json')):
        number = int(path.name.split('_')[1].split('.')[0])
        reservation = folder/'guard'/f'reservation_{number:06d}.json'
        if number in known or not reservation.exists():
            continue
        reserved = json.loads(reservation.read_text())
        result = json.loads(path.read_text())
        program = folder/reserved['path']
        if 'combined_score' not in result or not program.exists() or digest(program) != reserved['hash']:
            continue
        recs.append(dict(id=number, path=reserved['path'], hash=reserved['hash'],
                         score=result['combined_score'], result=result,
                         valid=result.get('n_failed', 0)==0, evaluation=number+1,
                         elapsed=result.get('completed_at', path.stat().st_mtime)-manifest['task_start'],
                         parent=None if number==0 else 'upstream', recovered=True))
    recs.sort(key=lambda r:r['id'])
    return manifest['config'], recs


def validate_run(folder, top, output, parallel, candidate_workers=1):
    folder = Path(folder).resolve()
    if parallel < 1 or candidate_workers < 1:
        raise ValueError('Validation worker counts must be positive')
    with (folder/'validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _validate_run(folder, top, output, parallel, candidate_workers)


def _validate_run(folder, top, output, parallel, candidate_workers):
    started = time.monotonic()
    config, archive = records(folder)
    if not archive:
        raise ValueError('No evaluated programs')
    signal_task = config['task'] == 'signal_processing'
    chosen = sorted((r for r in archive if not r['result'].get('outer_timeout')
                     and (not config.get('causality_policy') or r['result'].get('causality_passed'))),
                    key=lambda r: r['score'], reverse=True)
    if not chosen:
        raise ValueError('No uncensored training scores to validate')
    if top:
        chosen = chosen[:top]
    cache = folder/(f'validation_results_{CAUSALITY_POLICY}' if signal_task else 'validation_results')
    cache.mkdir(exist_ok=True)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    def validate_candidate(r):
        if digest(folder/r['path']) != r['hash']:
            raise ValueError('Logged program modified after discovery')
        path = cache/f"{r['id']:06d}.json"
        if path.exists():
            item = json.loads(path.read_text())
            if item.get('hash') != r['hash'] or item.get('task') != config['task']:
                raise ValueError('Validation cache provenance mismatch')
        else:
            score = evaluate(config['task'], folder/r['path'], 'val', timeout=3600, parallel=parallel,
                             work=cache/f"{r['id']:06d}.log")
            item = dict(id=r['id'], path=r['path'], train=r['score'], hash=r['hash'],
                        task=config['task'], validation=score['combined_score'], result=score)

            if not score.get('outer_timeout'):
                atomic_json(path,item)
        return item

    results = {}
    with ThreadPoolExecutor(max_workers=min(candidate_workers, len(chosen))) as pool:
        futures = {pool.submit(validate_candidate, record): index for index, record in enumerate(chosen)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()


            result = [results[index] for index in sorted(results)]
            scored = [item for item in result if not item['result'].get('outer_timeout')]
            eligible = [r for r in scored if not signal_task or r['result'].get('causality_passed')]
            best = max(eligible, key=lambda r:r['validation'], default=None)
            atomic_json(output,dict(task=config['task'],run=str(folder),test_only=config['test_only'],
                        selection_rule=f'top {top or "all"} by training, then best validation',
                        candidates=result,best=best,completed=len(scored)==len(chosen),
                        validation_wall_seconds=time.monotonic()-started,
                        seed=config['seed'], variant=config['variant'],
                        n_requested=len(chosen)))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evox = commands.add_parser('evox')
    evox.add_argument('--task', required=True, choices=TASKS)
    evox.add_argument('--seed', type=int, required=True)
    evox.add_argument('--stage', choices=('smoke', 'pilot', 'full', 'custom'), default='custom')
    evox.add_argument('--variant', choices=tuple(VARIANTS), default='evox_lineage_reward')
    evox.add_argument('--evaluations', type=int, default=500)
    evox.add_argument('--seconds', type=float)
    evox.add_argument('--deadline', type=float)
    evox.add_argument('--parallel', type=int, default=1)
    evox.add_argument('--upstream', required=True)
    evox.add_argument('--api-base', required=True)
    evox.add_argument('--model', default='vllm/qwen3-coder-30b-a3b')
    evox.add_argument('--server-manifest', required=True)
    evox.add_argument('--output', required=True)
    validation = commands.add_parser('validate')
    validation.add_argument('folder')
    validation.add_argument('--top', type=int, default=20)
    validation.add_argument('--parallel', type=int, default=1)
    validation.add_argument('--candidate-workers', type=int, default=1)
    validation.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'evox':
        launch(args)
    else:
        if args.top < 0:
            parser.error('--top must be nonnegative')
        print(json.dumps(validate_run(args.folder, args.top, args.output,
                                      args.parallel, args.candidate_workers), indent=2))


if __name__ == '__main__':
    main()
