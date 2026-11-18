"""Prepare the local Qwen cache used by the runner."""
import argparse
import json
from pathlib import Path


MODEL = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
REVISION = 'b2cff646eb4bb1d68355c01b18ae02e7cf42d120'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--local-only', action='store_true', help='Verify existing weights without downloading')
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    marker = destination / 'ahd-model.json'
    if args.local_only:
        from register_local import register
        record = register(destination)
        if record.get('upstream_revision') and record['upstream_revision'] != REVISION:
            raise SystemExit('Local weights belong to a different model revision.')
        return
    identity = {'model': MODEL, 'revision': REVISION}
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if any(recorded.get(key) != value for key, value in identity.items()):
            raise SystemExit('The destination contains a different model revision. Choose another directory.')
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=REVISION, local_dir=str(destination))
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print(f'Model ready at {destination}')


if __name__ == '__main__':
    main()
