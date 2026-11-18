import argparse
import json
from pathlib import Path
import subprocess
from ahd.protocol import MODEL, atomic_json

p=argparse.ArgumentParser()
p.add_argument('--model-dir',required=True)
p.add_argument('--output',required=True)
p.add_argument('--tensor-parallel',type=int,required=True)
p.add_argument('--port',type=int,required=True)
a=p.parse_args()
record=json.loads((Path(a.model_dir)/'ahd-model.json').read_text())
assert record['model']==MODEL
atomic_json(a.output,dict(model=MODEL,revision=record['revision'],dtype='bfloat16',
    quantization=None,max_model_len=16384,served_model_name='vllm/qwen3-coder-30b-a3b',
    model_directory=str(Path(a.model_dir).resolve()),tensor_parallel_size=a.tensor_parallel,
    serving_command=['vllm','serve',a.model_dir,'--host','127.0.0.1','--port',str(a.port),
      '--dtype','bfloat16','--max-model-len','16384','--served-model-name',
      'vllm/qwen3-coder-30b-a3b','--generation-config','vllm',
      '--tensor-parallel-size',str(a.tensor_parallel)],
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                                  '--format=csv,noheader'],text=True)))
