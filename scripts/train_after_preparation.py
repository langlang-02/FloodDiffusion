#!/usr/bin/env python3
"""Wait for a specific live preparer, then run the verified 500-epoch pipeline."""
import argparse
import json
import os
from pathlib import Path
import time


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0], fields[19]  # state and kernel start time; guards PID reuse
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-pid',type=int,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    data_root=args.data_root.resolve()
    identity=process_identity(args.prepare_pid)
    if identity is not None:
        command=Path(f'/proc/{args.prepare_pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
        if 'prepare_g1.py' not in command:
            raise ValueError('Specified PID is not the G1 preparer')
        print(f'Waiting for preparer PID={args.prepare_pid}, start={identity[1]}, data={data_root}',flush=True)
        tick=0
        while True:
            current=process_identity(args.prepare_pid)
            if current is None or current[0]=='Z' or current[1]!=identity[1]:
                break
            if tick and tick%12==0:
                print(f'Preparer PID={args.prepare_pid} is still live; training has not started',flush=True)
            time.sleep(5)
            tick+=1
    manifest=json.loads((data_root/'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('smoke'):
        raise RuntimeError('Preparer stopped without a complete production dataset; refusing to train')
    print('Production preparation complete; launching source verification, full audit and 500-epoch training',flush=True)
    os.chdir(root)
    env=dict(os.environ,G1_DATA_ROOT=str(data_root))
    os.execvpe('bash',['bash',str(root/'scripts/run_g1_500epochs.sh')],env)


if __name__=='__main__':main()
