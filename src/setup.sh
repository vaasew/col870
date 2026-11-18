#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python="${PYTHON:-python3.12}"
"$python" -m venv .envs/run
.envs/run/bin/python -m pip install --upgrade pip
.envs/run/bin/python -m pip install -r requirements.txt
.envs/run/bin/python -m pip install --no-deps -e vendor/skydiscover
if [[ "${1:-}" == "--serve" ]]; then
  "$python" -m venv .envs/serve
  .envs/serve/bin/python -m pip install --upgrade pip
  .envs/serve/bin/python -m pip install vllm huggingface_hub
fi
mkdir -p environment
.envs/run/bin/python -m pip freeze > environment/run.txt
if [[ -x .envs/serve/bin/python ]]; then
  .envs/serve/bin/python -m pip freeze > environment/serve.txt
fi
.envs/run/bin/python check.py
