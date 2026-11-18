"""Run the frozen discovery method and validate its saved shortlist."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ahd.cli import validate_run
from ahd.protocol import ROOT, TASKS, wall_limit


def cpu_count():
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    allocation = int(os.environ.get('AHD_CPU_LIMIT', affinity))
    return max(1, min(affinity, allocation))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--server-manifest', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--discovery-only', action='store_true')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    cpus = min(cpu_count(), args.workers if args.workers is not None else cpu_count())
    if cpus < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    manifest = args.server_manifest.resolve() if args.server_manifest else None
    if not args.validate_only and manifest is None:
        parser.error('--server-manifest is required for discovery')
    if args.validate_only and args.discovery_only:
        parser.error('Choose either discovery-only or validate-only')
    policy = json.loads((ROOT / 'frozen.json').read_text())
    discount = policy['lambda'][args.task]
    variant = 'evox_lineage_memory' if discount == 1 else 'evox_lineage_reward'
    instances = 1 if args.task in TASKS[-2:] else cpus
    if not args.validate_only:
        deadline = args.deadline or time.time() + wall_limit(args.task)
        command = [sys.executable, '-m', 'ahd.cli', 'evox', '--task', args.task,
                   '--seed', str(args.seed), '--stage', 'full', '--variant', variant,
                   '--evaluations', '500', '--seconds', str(wall_limit(args.task)),
                   '--deadline', str(deadline), '--parallel', str(instances),
                   '--upstream', str(ROOT / 'vendor/skydiscover'),
                   '--api-base', args.api_base, '--server-manifest', str(manifest),
                   '--output', str(output)]
        discovery = subprocess.run(command, cwd=ROOT)
        status = discovery.returncode
    else:
        status = 0
    if args.discovery_only:
        raise SystemExit(status)
    if not (output / 'manifest.json').is_file():
        raise SystemExit(status or 'Discovery manifest is missing')
    validation_instances = min(4, instances)
    selected = validate_run(output, 20, output / 'selection.json', validation_instances,
                            max(1, cpus // validation_instances))
    print(json.dumps(selected, indent=2))
    if status:
        print(f'Discovery exited {status}; saved candidates were validated.', file=sys.stderr)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
