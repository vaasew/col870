from pathlib import Path
import hashlib
import os
import json
import math

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = Path(os.environ.get('AHD_PROBLEMS_DIR', ROOT.parent / 'problems')).resolve()

TASKS = ('prism', 'cloudcast', 'signal_processing', 'heilbronn_triangle', 'circle_packing_rect')
INSTANCE_SECONDS = dict(zip(TASKS, (10,60,10,360,360)))
MODEL = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
DECODING = dict(temperature=0.99, top_p=0.95, max_tokens=8000)
ENTRIES = dict(zip(TASKS, ('compute_model_placement', 'search_algorithm',
                         'run_signal_processing', 'heilbronn_triangle11', 'circle_packing21')))
EDITABLE_ENTRIES = {**ENTRIES, 'signal_processing': 'process_signal'}


def task_path(task):
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    return PROBLEMS / task


def wall_limit(task):
    return 36000 if task in TASKS[-2:] else 18000


def score_profile(result):
    return tuple(round(row['score'], 9) for batch in result.get('instance_batches', [])
                 for row in batch['results'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        import os
        os.fsync(f.fileno())
    temp.replace(path)


def validate_config(c):
    task_path(c['task'])
    if not 1 <= c['evaluations'] <= 500:
        raise ValueError('Evaluations must be between 1 and 500, including initial evaluation')
    if not 0 < c['seconds'] <= wall_limit(c['task']):
        raise ValueError('Wall budget exceeds protocol or is nonpositive')
    if not math.isfinite(c['seconds']):
        raise ValueError('Nonfinite wall budget')
    if c['parallel'] < 1:
        raise ValueError('parallel must be positive')
    if not 0 <= c['gamma'] <= 1:
        raise ValueError('gamma must be in [0,1]')
