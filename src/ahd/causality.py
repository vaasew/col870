"""Prefix invariance checks for the course signal entry point."""
import random
from types import SimpleNamespace

import numpy as np

CAUSALITY_POLICY = 'prefix-v1'


def install(mod):
    original = mod._one
    load_program = mod.common.load_program

    def checked(program, signal):
        def capture(path):
            entry = load_program(path).run_signal_processing

            def run(**kwargs):
                before = random.getstate(), np.random.get_state()
                result = entry(**kwargs)
                captured.update(entry=entry, before=before, result=result)
                return result

            return SimpleNamespace(run_signal_processing=run)

        captured = {}
        mod.common.load_program = capture
        try:
            result = original(program, signal)
        finally:
            mod.common.load_program = load_program
        if result['program_time'] > mod.TIME_LIMIT:
            return result
        noisy, window = signal['noisy'], mod.WINDOW
        full = np.asarray(captured['result']['filtered_signal']).ravel()
        if len(full) != len(noisy)-window+1:
            raise ValueError('Causality check failed: expected one aligned output per complete window')
        after = random.getstate(), np.random.get_state()
        try:
            for end in sorted({max(window, len(noisy)*k//4) for k in (1, 2, 3)}):
                random.setstate(captured['before'][0])
                np.random.set_state(captured['before'][1])

                entry = load_program(program).run_signal_processing
                random.setstate(captured['before'][0])
                np.random.set_state(captured['before'][1])
                try:
                    prefix = np.asarray(entry(noisy_signal=noisy[:end].copy(),
                                              window_size=window)['filtered_signal']).ravel()
                except Exception as exc:
                    raise ValueError(f'Causality check failed: prefix {end} raised {type(exc).__name__}') from exc
                expected = full[:end-window+1]
                if prefix.shape != expected.shape or not np.allclose(prefix, expected, rtol=1e-8, atol=1e-10):
                    raise ValueError(f'Causality check failed at input prefix {end}')
        finally:
            random.setstate(after[0])
            np.random.set_state(after[1])
        return result

    mod._one = checked
