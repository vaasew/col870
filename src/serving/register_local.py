"""Register already downloaded BF16 weights locally, without network/model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ahd.protocol import MODEL, atomic_json


def register(root):
    root=Path(root).expanduser().resolve()
    config=json.loads((root/'config.json').read_text())
    if config.get('model_type')!='qwen3_moe' or config.get('quantization_config'):
        raise ValueError('Expected unquantized Qwen3 MoE config')
    index=json.loads((root/'model.safetensors.index.json').read_text())
    shards=sorted(set(index['weight_map'].values()))
    if not shards:raise ValueError('No indexed weights')
    for tokenizer in ('tokenizer.json','tokenizer_config.json'):
        if not (root/tokenizer).is_file():raise ValueError(f'Missing {tokenizer}')
    hashes={}
    for filename in shards:
        path=root/filename
        if path.parent!=root or not path.is_file():raise ValueError(f'Missing/unsafe shard {filename}')
        with path.open('rb') as f:
            header_size=struct.unpack('<Q',f.read(8))[0]
            if header_size>128*1024*1024:raise ValueError('Invalid safetensors header')
            header=json.loads(f.read(header_size))
            for key,tensor in header.items():
                if key=='__metadata__':continue
                if tensor['dtype'] not in ('BF16','I64','I32','U8','BOOL'):
                    raise ValueError(f'Unexpected dtype {tensor["dtype"]}: {filename}/{key}')
                if 8+header_size+tensor['data_offsets'][1]>path.stat().st_size:
                    raise ValueError(f'Truncated shard: {filename}')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        hashes[filename]=h.hexdigest()
        print('Verified BF16 shard',filename,flush=True)
    for filename in ('config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'):
        hashes[filename]=hashlib.sha256((root/filename).read_bytes()).hexdigest()
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    revisions=set()
    for path in (root/'.cache/huggingface/download').glob('*.metadata'):
        lines=path.read_text().splitlines()
        if lines and re.fullmatch('[0-9a-f]{40}',lines[0]):revisions.add(lines[0])
    upstream_revision=next(iter(revisions)) if len(revisions)==1 else None
    # Never invent an upstream commit if the download metadata was not retained.
    record=dict(model=MODEL,revision=upstream_revision or 'local-sha256:'+fingerprint,
                upstream_revision=upstream_revision,content_fingerprint=fingerprint,
                revision_origin='huggingface_metadata' if upstream_revision else 'local_content_sha256',
                dtype='bfloat16',files=hashes)
    atomic_json(root/'ahd-model.json',record)
    print('Registered cached model:',root,flush=True)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True)
    register(p.parse_args().destination)
