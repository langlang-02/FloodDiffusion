#!/usr/bin/env python3
"""Launch a trainer and mirror its epoch metrics/evaluation to one W&B run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import wandb


def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def flatten(value,prefix=''):
    result={}
    for key,item in value.items():
        name=f'{prefix}/{key}' if prefix else key
        if isinstance(item,dict):result.update(flatten(item,name))
        elif isinstance(item,(int,float,bool,str)):result[name]=item
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--project',default=os.getenv('WANDB_PROJECT','flood-g1-vae'))
    parser.add_argument('--entity',default=os.getenv('WANDB_ENTITY'))
    parser.add_argument('--mode',choices=['online','offline'],default=os.getenv('WANDB_MODE','online'))
    parser.add_argument('--poll-seconds',type=float,default=5.)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:parser.error('Missing training command after --')
    output=args.output.resolve()
    tracking=output.parent/'.wandb_tracking'/output.name
    tracking.mkdir(parents=True,exist_ok=True)
    info_path=tracking/'tracking.json';previous=read_json(info_path) or {}
    if previous and (previous['project']!=args.project or previous['mode']!=args.mode):raise ValueError('Tracking configuration changed; use a new run directory')
    run=wandb.init(project=args.project,entity=args.entity,mode=args.mode,
                   id=previous.get('id') or wandb.util.generate_id(),resume='allow' if args.mode=='online' else None,
                   name=output.name,dir=str(tracking),job_type='training',save_code=False,
                   config={'output_directory':str(output),'metrics_source':'trainer metrics.jsonl; unweighted loss components'})
    info=dict(id=run.id,url=run.url,project=args.project,entity=run.entity,mode=args.mode,last_epoch=previous.get('last_epoch',0))
    def save():
        temporary=info_path.with_suffix('.tmp');temporary.write_text(json.dumps(info,indent=2));temporary.replace(info_path)
    save();print(f'W&B tracking: {run.url or tracking} (mode={args.mode})',flush=True)
    run.define_metric('optimizer_step');run.define_metric('epoch')
    for pattern in ('train/*','val/*','train_weighted/*','val_weighted/*','lr','epoch_seconds','peak_memory_gib','fixed_val/*','selection/*'):
        run.define_metric(pattern,step_metric='optimizer_step')
    identity_seen=None;evaluations=set()
    def sync():
        nonlocal identity_seen
        identity=read_json(output/'identity.json')
        if identity is not None and identity!=identity_seen:
            run.config.update(identity,allow_val_change=True);identity_seen=identity
        path=output/'metrics.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:record=json.loads(line)
                except json.JSONDecodeError:continue
                if record['epoch']<=info['last_epoch']:continue
                values=flatten(record);values['optimizer_step']=record['step']
                if 'weights' in (identity_seen or {}):
                    for phase in ('train','val'):
                        for key,weight in identity_seen['weights'].items():
                            if key in record.get(phase,{}):values[f'{phase}_weighted/{key}']=record[phase][key]*weight
                run.log(values);info['last_epoch']=record['epoch'];save()
        status=read_json(output/'status.json')
        if status:run.summary.update({f'status/{k}':v for k,v in status.items() if k not in ('rank_pids','pid')})
        for name in ('best','latest'):
            result=read_json(output/f'eval_{name}/summary.json')
            if name not in evaluations and result and result.get('complete'):
                values=flatten(result,f'test/{name}');run.log(values);run.summary.update(values);evaluations.add(name)
    child=None
    def forward(signum,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signum)
    signal.signal(signal.SIGTERM,forward);signal.signal(signal.SIGINT,forward)
    code=1
    try:
        child=subprocess.Popen(command,start_new_session=True)
        while child.poll() is None:
            sync();time.sleep(args.poll_seconds)
        code=child.returncode;sync();run.summary['trainer_exit_code']=code
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        run.finish(exit_code=code)
    raise SystemExit(code)

if __name__=='__main__':main()
