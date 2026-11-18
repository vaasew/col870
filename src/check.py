"""Check required task files before starting discovery."""
from ahd.protocol import PROBLEMS, TASKS

for path in [PROBLEMS / 'evalkit.py', *(
    PROBLEMS / task / name
    for task in TASKS
    for name in ('initial_program.py', 'evaluator.py', 'prompt.txt', 'config.yaml')
)]:
    if not path.is_file():
        raise SystemExit(f'Missing required file: {path}')
print('Task files available.')
