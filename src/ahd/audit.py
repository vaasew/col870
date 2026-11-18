"""Portable provenance and durable, append-only lifecycle records."""
import json
from pathlib import Path
import os
import time
import traceback
from .protocol import atomic_json


def event(folder, kind, **fields):
    row = {**fields, 'time':fields.get('time',time.time()), 'kind':kind}
    with (Path(folder)/'events.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def failure(folder, exc, **fields):
    value = dict(error=str(exc), type=type(exc).__name__, traceback=traceback.format_exc(),
                 time=time.time(), **fields)
    atomic_json(Path(folder)/'failure.json', value)
    event(folder, 'failure', **value)
