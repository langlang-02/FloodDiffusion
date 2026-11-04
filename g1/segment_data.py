"""Auditable source-time to latent-time alignment for segment-conditioned G1."""
import json,math
from collections import Counter
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
from g1.common import sha256,write_json

def validate_events(events):
    result=[]
    for event in events:
        start,end=float(event['start_time']),float(event['end_time'])
        text=event['description']
        if not isinstance(text,str) or not text.strip() or not math.isfinite(start+end) or start<0 or end<=start:
            raise ValueError(f'Invalid temporal event: {event}')
        result.append((start,end,text.strip()))
    result.sort(key=lambda x:(x[0],x[1]))
    if any(a[1]>b[0] for a,b in zip(result,result[1:])):
        raise ValueError('Overlapping temporal events')
    return result

def align_events(row,events):
    length=int(row['frames']);start=int(row['start_frame'])
    if length<1 or start<0:raise ValueError('Invalid motion window')
    latent_length=1+(length-1+3)//4
    times=(start+np.minimum(np.arange(latent_length)*4,length-1))/20.
    labels=['']*latent_length;covered=0;short=0
    for left,right,text in events:
        mask=(times>=left)&(times<right)
        for j in np.flatnonzero(mask):labels[int(j)]=text
        covered+=int(mask.sum())
        if left<(start+length)/20 and right>start/20 and not mask.any():short+=1
    texts=[];ends=[]
    for j,text in enumerate(labels):
        if not texts or text!=texts[-1]:texts.append(text);ends.append(j+1)
        else:ends[-1]=j+1
    return dict(text=texts,token_text_end=ends,latent_length=latent_length,covered=covered,short_events=short)

def prepare(native_root,temporal_path,output):
    native_root,output=Path(native_root),Path(output)
    manifest=json.loads((native_root/'manifest.json').read_text())
    if sha256(temporal_path)!=manifest['temporal_sha256']:raise ValueError('Temporal source hash changed')
    temporal={}
    for line in Path(temporal_path).read_text().splitlines():
        item=json.loads(line);name=item['filename']
        if name in temporal:raise ValueError(f'Duplicate temporal source {name}')
        temporal[name]=validate_events(item['events'])
    labels={};stats={s:Counter() for s in ('train','val','test')};excluded=[];kept=[]
    for line in (native_root/'index.jsonl').read_text().splitlines():
        row=json.loads(line);source=Path(row['source']).stem
        if source not in temporal:raise ValueError(f'Missing temporal source {source}')
        record=align_events(row,temporal[source]);labels[row['id']]=record
        st=stats[row['split']];st['windows']+=1;st['latent_positions']+=record['latent_length']
        st['gap_positions']+=record['latent_length']-record['covered'];st['short_events']+=record['short_events']
        st['multi_label_windows']+=len(set(t for t in record['text'] if t))>1
        st['unlabelled_windows']+=record['covered']==0
        if row['split']=='train':
            (kept if record['covered'] else excluded).append(row['id'])
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'labels.json',labels)
    (output/'train.txt').write_text('\n'.join(kept)+'\n');(output/'excluded_train.txt').write_text('\n'.join(excluded)+'\n')
    write_json(output/'manifest.json',dict(complete=True,native_manifest_sha256=sha256(native_root/'manifest.json'),
        temporal_sha256=sha256(temporal_path),labels_sha256=sha256(output/'labels.json'),train_sha256=sha256(output/'train.txt'),
        excluded_sha256=sha256(output/'excluded_train.txt'),adapter_sha256=sha256(__file__),stats=stats,
        training_samples=len(kept),excluded_training_samples=len(excluded),
        time_mapping='source seconds = (start_frame + min(4*j,frames-1))/20; left closed/right open',
        gap_policy='empty text; keep motion supervision',exclusion_policy='exclude train windows without any labelled latent position'))
    return stats

def load_labels(directory,native_root):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    assert m['complete'] and m['native_manifest_sha256']==sha256(Path(native_root)/'manifest.json')
    assert m['adapter_sha256']==sha256(__file__)
    for file,key in [('labels.json','labels_sha256'),('train.txt','train_sha256'),('excluded_train.txt','excluded_sha256')]:
        if sha256(directory/file)!=m[key]:raise ValueError(f'Changed segment data: {file}')
    return json.loads((directory/'labels.json').read_text()),set((directory/'train.txt').read_text().splitlines()),m

class SegmentMotions(Dataset):
    def __init__(self,rows,rot_root,labels,training=False):
        self.rows,self.root,self.labels=rows,Path(rot_root),labels
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];x=np.load(self.root/'motions'/f"{row['id']}.npy")
        if len(x)!=row['frames']:raise ValueError('Motion/index length mismatch')
        return torch.from_numpy(x),self.labels[row['id']],row

def conditions(items):
    return dict(text=[r['text'] for r in items],feature_text_end=[r['token_text_end'] for r in items])
