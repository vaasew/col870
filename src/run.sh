#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${1:?Usage: bash src/run.sh TASK SEED [OUTPUT]}"
task="$1"
started_epoch=$(date +%s)
seed="${2:-0}"
output="${3:-$PWD/../runs/$task/seed$seed/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$output"
output=$(cd "$output" && pwd)
python="${RUN_PYTHON:-$PWD/.envs/run/bin/python}"
serve="${SERVE_COMMAND:-$PWD/.envs/serve/bin/vllm}"
model="${MODEL_DIR:-$PWD/../models/Qwen3-Coder}"
export PYTHONPATH="$PWD:$PWD/vendor/skydiscover${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export VLLM_USE_FLASHINFER_SAMPLER=0
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
"$python" check.py
"$python" - "$model" <<'PY'
import json, sys
from pathlib import Path
from ahd.protocol import MODEL
identity = json.loads((Path(sys.argv[1]) / 'ahd-model.json').read_text())
revision = identity.get('revision', '')
local = revision.startswith('local-sha256:') and identity.get('dtype') == 'bfloat16'
if identity.get('model') != MODEL or not (revision == 'b2cff646eb4bb1d68355c01b18ae02e7cf42d120' or local):
    raise SystemExit('Qwen cache identity differs from the frozen revision')
PY
budget=$("$python" -c 'import sys; from ahd.protocol import wall_limit; print(wall_limit(sys.argv[1]))' "$task")
deadline=$(( started_epoch + budget ))
port=$("$python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
server=
cleanup() {
  if [[ -n "$server" ]]; then
    kill -TERM -- "-$server" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
setsid "$serve" serve "$model" --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --max-model-len 16384 --served-model-name vllm/qwen3-coder-30b-a3b \
  --generation-config vllm --tensor-parallel-size 1 > "$output/server.log" 2>&1 &
server=$!
"$python" serving/wait_server.py --port "$port" --pid "$server" --timeout 1800
"$python" serving/write_manifest.py --model-dir "$model" --output "$output/server.json" \
  --tensor-parallel 1 --port "$port"
discovery_status=0
"$python" run.py --discovery-only --task "$task" --seed "$seed" --output "$output/discovery" \
  --api-base "http://127.0.0.1:$port/v1" --server-manifest "$output/server.json" --deadline "$deadline" || discovery_status=$?
cleanup
server=
if [[ "${VALIDATE_AFTER:-1}" == 1 ]]; then
  "$python" run.py --validate-only --task "$task" --seed "$seed" --output "$output/discovery"
fi
exit "$discovery_status"
