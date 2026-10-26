"""Detached supervisor: wait for verified data, train two four-GPU jobs, evaluate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.g1.common import atomic_json


def run_variant(args, name, gpus):
    output = args.run / name
    output.mkdir(exist_ok=True)
    status_path = output / 'status.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', PYTHONUNBUFFERED='1')
    def status(stage, **extra):
        atomic_json(status_path, dict(stage=stage, timestamp=time.time(), gpus=gpus, **extra))
    try:
        status('waiting_for_data')
        deadline = time.monotonic() + args.wait_hours * 3600
        while not (args.data / 'VERIFIED.json').is_file():
            if time.monotonic() > deadline:
                raise TimeoutError('Dataset did not become ready within the configured wait period')
            time.sleep(60)
        status('waiting_for_gpus')
        while True:
            result = subprocess.check_output(['nvidia-smi', f'--id={gpus}',
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            usage = [[int(v.strip()) for v in line.split(',')] for line in result.strip().splitlines()]
            if len(usage) == 4 and all(memory < 1024 and utilization < 5 for memory, utilization in usage):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('GPU group did not become idle')
            time.sleep(60)
        config = args.source / 'configs' / f'g1_{name}.yaml'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   str(args.source / 'scripts/g1/train.py'), '--config', str(config),
                   '--data', str(args.data), '--output', str(output)]
        if args.resume and (output / 'last.pt').exists():
            command.append('--resume')
        atomic_json(output / 'commands.json', {'train': command})
        status('training', command=command)
        with (output / 'train.log').open('a') as log:
            subprocess.run(command, env=env, cwd=args.source, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not (output / 'training_complete.json').is_file():
            raise RuntimeError('Training exited without completion marker')
        status('evaluating')
        command = [sys.executable, str(args.source / 'scripts/g1/evaluate.py'),
                   '--checkpoint', str(output / 'best.pt'), '--data', str(args.data),
                   '--mjcf', str(args.mjcf), '--output', str(output / 'evaluation')]
        with (output / 'evaluation.log').open('a') as log:
            subprocess.run(command, env=env, cwd=args.source, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((output / 'evaluation/metrics.json').read_text())
        if not metrics['complete'] or metrics['partial'] or metrics['smoke_checkpoint']:
            raise RuntimeError('Evaluation is incomplete or a smoke run')
        status('complete', metrics=metrics['metrics'])
    except Exception as error:
        status('failed', error=str(error))
        print(name, 'FAILED:', error, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--mjcf', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--wait-hours', type=float, default=24)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.run = args.run.resolve(); args.data = args.data.resolve(); args.mjcf = args.mjcf.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    lock = (args.run / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.run / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    root = Path(__file__).resolve().parents[2]
    args.source = args.run / 'source'
    if not args.source.exists():
        args.source.mkdir()
        for name in ('scripts', 'models', 'vendor', 'configs'):
            shutil.copytree(root / name, args.source / name, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('LICENSE', 'THIRD_PARTY_LICENSES.md'):
            shutil.copy2(root / name, args.source / name)
    elif not args.resume:
        raise ValueError('Run already has a source snapshot; use a fresh run or --resume')
    threads = [threading.Thread(target=run_variant, args=(args, name, gpus))
               for name, gpus in [('rot6d68', '0,1,2,3'), ('sincos66', '4,5,6,7')]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    results = {name: json.loads((args.run / name / 'status.json').read_text()) for name in ('rot6d68', 'sincos66')}
    atomic_json(args.run / 'summary.json', results)
    if any(result['stage'] != 'complete' for result in results.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
