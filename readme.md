GitHub repository: https://github.com/vaasew/col870

Siddharthan Dileep — Kerberos `aiy257585`.

## Requirements

Run commands from the repository root. Task files are read from
`problems/<task>/`, alongside `src/`.

## Environment setup

```bash
bash src/setup.sh --serve
```

This creates `src/.envs/run` for discovery and evaluation and
`src/.envs/serve` for vLLM. Installed versions are recorded in
`src/environment/`. Set `PYTHON` to override the Python 3.12 executable.

## Qwen setup

**Download the model**

```bash
export MODEL_DIR="$PWD/models/Qwen3-Coder"
src/.envs/serve/bin/python src/serving/download_model.py \
  --destination "$MODEL_DIR"
```

**Use existing local weights**

```bash
export MODEL_DIR="/absolute/path/to/Qwen3-Coder"
src/.envs/serve/bin/python src/serving/download_model.py \
  --destination "$MODEL_DIR" --local-only
```

Local preparation verifies the configuration, tokenizer and BF16 weight
shards and records their hashes. If `ahd-model.json` already exists from
preparation, setting `MODEL_DIR` is sufficient. When revision metadata is
absent, the local cache is identified by its content fingerprint.

**Qwen runs locally through vLLM on the same machine.** The runner loads
`Qwen/Qwen3-Coder-30B-A3B-Instruct` from `MODEL_DIR` and sends requests to
`127.0.0.1`. `/tokenize` counts rendered tokens and `/v1/chat/completions`
generates responses. Generation and meta-search share the server.

Settings are BF16 without quantization, context length 16,384, temperature
0.99, top-p 0.95, and at most 8,000 output tokens per request.

## Run

```bash
bash src/run.sh signal_processing 0
```

Arguments are task, seed, and an optional output directory:

```bash
bash src/run.sh heilbronn_triangle 1 "$PWD/runs/heilbronn-seed1"
```

Tasks are `prism`, `cloudcast`, `signal_processing`, `heilbronn_triangle`,
and `circle_packing_rect`.

The runner starts vLLM, performs discovery, stops vLLM, and validates up to
20 training-shortlisted programs on CPUs. Each evaluator worker uses one
BLAS thread. Set `AHD_CPU_LIMIT` to bound CPU workers within the available
affinity.

Environment overrides:

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_DIR` | Local Qwen weights | `models/Qwen3-Coder` |
| `RUN_PYTHON` | Runner Python executable | `src/.envs/run/bin/python` |
| `SERVE_COMMAND` | vLLM executable | `src/.envs/serve/bin/vllm` |
| `AHD_CPU_LIMIT` | Maximum CPU workers | Process CPU affinity |

## Method settings

EvoX strategy evolution and native variation are combined with editable
blocks, fixed-four Bandit scheduling, immediate lineage selection, and
ranked summary memory capped at 384 rendered tokens.

Lineage reward is `lambda * (g - h) + h`. Here `g` is the positive
child-over-parent gain, and `h` is the portion above the pre-edit incumbent.

| Task | Lambda | Maximum evaluations | Discovery limit |
|---|---:|---:|---:|
| PRISM | 0.25 | 500 | 5 hours |
| Cloudcast | 1.0 | 500 | 5 hours |
| Signal processing | 1.0 | 500 | 5 hours |
| Heilbronn triangle | 1.0 | 500 | 10 hours |
| Rectangle circle packing | 0.25 | 500 | 10 hours |

`src/frozen.json` records the settings. Discovery time includes model
startup. Failed evaluations consume slots. Signal search and validation
enforce prefix invariance.

## Evaluate a selected program

```bash
src/.envs/run/bin/python problems/prism/evaluator.py \
  leaderboard3/prism/2025AIY7585_heuristic.py --split val
```

Programs are at `leaderboard3/<task>/2025AIY7585_heuristic.py`.

## Outputs and recovery

Outputs default to `runs/TASK/seedSEED/RUN_ID/`.

| File | Contents |
|---|---|
| `server.log`, `server.json` | Serving logs and model configuration |
| `discovery/manifest.json` | Run configuration and budgets |
| `discovery/events.jsonl` | Calls, tokens, strategy actions, parent choices and memory delivery |
| `discovery/guard/` | Programs, evaluation reservations, scores, per-instance outcomes and timings |
| `discovery/selection.json` | Selected program and validation score |

To validate saved candidates:

```bash
src/.envs/run/bin/python src/run.py --validate-only \
  --task signal_processing --seed 0 \
  --output runs/signal_processing/seed0/RUN_ID/discovery
```
