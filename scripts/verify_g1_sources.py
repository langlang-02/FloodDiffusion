#!/usr/bin/env python3
"""Verify the existing authorized BOS download without rewriting raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=Path('/data/langlang/dataset/supermimic/raw/bones_seed'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.raw_root/'source_manifest.json').read_text())
    results=[]
    started=time.time()
    for item in manifest['files']:
        path=(args.raw_root/item['path']).resolve()
        if not path.is_relative_to(args.raw_root.resolve()) or path.stat().st_size!=item['bytes']:
            raise ValueError(f'Invalid path/size: {item["path"]}')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(16*1024*1024),b''):
                digest.update(block)
        if digest.hexdigest()!=item['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        results.append(item)
        print(f'Verified {item["path"]}',flush=True)
    actual=sum(1 for _ in (args.raw_root/'g1').rglob('*.csv'))
    extraction=json.loads((args.raw_root/'extraction.json').read_text())
    if actual!=extraction['files']:
        raise ValueError(f'Extracted CSV count mismatch: {actual}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(complete=True,files=results,csv_count=actual,
        raw_root=str(args.raw_root.resolve()),seconds=time.time()-started),indent=2)+'\n')


if __name__=='__main__':main()
