#!/usr/bin/env python3
"""Reaggregate exact per-clip metrics using previously audited tracker filters."""
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/bones_g1_native38'
OLD=Path('/data/langlang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1/flat_ground_v1')
META=Path('/data/langlang/dataset/supermimic/raw/bones_seed/metadata/seed_metadata_v004.csv')
SUPER=Path('/data/langlang/dataset/supermimic/manifests/bones_seed_clip_identity.csv')
CODE=ROOT.parent/'momask-codes/dataset/bones-seed_prepare/g1_filter_flat_ground.py'
OUT=ROOT/'outputs/tracker_filtered_evaluation_v1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 OUT.mkdir(exist_ok=False)
 report=json.loads((OLD/'report.json').read_text())
 for path in (META,CODE):
  expected=[v for k,v in report['sha256'].items() if Path(k).name==path.name]
  assert len(expected)==1 and sha(path)==expected[0],path
 assert sha(SUPER)==json.loads(SUPER.with_suffix('.json').read_text())['manifest_sha256']
 spec=importlib.util.spec_from_file_location('tracker_filter',CODE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 test=[json.loads(l) for l in (DATA/'index.jsonl').read_text().splitlines() if json.loads(l)['split']=='test']
 sources={r['source'] for r in test};decisions={}
 with META.open() as f:
  for row in csv.DictReader(f):
   source=row['move_g1_path']
   if source not in sources:continue
   assert source not in decisions
   fields={k:v for k,v in row.items() if v and ('desc' in k or k=='content_name')}
   fields['source_motion_name']=row['filename'].split('__')[0]
   evidence=mod.classify({k:v for k,v in fields.items() if k!='content_natural_desc_4'})
   review=[] if evidence else mod.classify({'content_natural_desc_4':fields.get('content_natural_desc_4','')})
   decisions[source]=dict(keep=not evidence,evidence=evidence,review=review)
 assert set(decisions)==sources
 archived={json.loads(l)['source'] for l in (OLD/'excluded.jsonl').read_text().splitlines()}
 for source in sources&archived:assert not decisions[source]['keep'],source
 with SUPER.open() as f:super_decisions={r['source_relpath']:r for r in csv.DictReader(f)}
 audit=[]
 for r in test:
  d=super_decisions[r['source']];assert d['source_sha256']==r['source_sha256'],r['source']
  audit.append(dict(sample_id=r['id'],source=r['source'],frames=r['frames'],
   foot_contact_eligible=r['foot_contact_eligible'],mimiclite_keep=decisions[r['source']]['keep'],
   mimiclite_reasons=sorted({e['reason'] for e in decisions[r['source']]['evidence']}),
   mimiclite_review=bool(decisions[r['source']]['review']),supermimic_keep=d['keep']=='1',supermimic_reason=d['reason']))
 with (OUT/'filter_audit.jsonl').open('w') as f:
  for r in audit:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 datasets={'all':audit,'mimiclite_flat_ground_v1':[r for r in audit if r['mimiclite_keep']],
           'supermimic_tracker_kept':[r for r in audit if r['supermimic_keep']]}
 runs={'rot6d68_best':ROOT/'outputs/g1_rot6d68_textop_500ep_bs2048/eval_best',
       'native38_best':ROOT/'outputs/g1_causal_textop_500ep/eval_best',
       'rvq_rot6d68_ep499':ROOT/'outputs/vae_rvq_comparison/rvq_same_test'}
 metrics=['global_mpjpe_mm','root_relative_mpjpe_mm'];tables=[];identities={};summaries={}
 for label,selected in datasets.items():
  directory=OUT/label;directory.mkdir();ids=[r['sample_id'] for r in selected]
  (directory/'samples.txt').write_text('\n'.join(ids)+'\n')
  for model,run in runs.items():
   records={r['sample_id']:r for r in csv.DictReader((run/'per_sample.csv').open())}
   assert set(records)=={r['id'] for r in test}
   identity=json.loads((run/'identity.json').read_text());identities[model]=identity
   assert identity['split_sha256']==sha(DATA/'test.txt') and identity['mjcf_sha256']==sha(DATA/'kinematic.xml')
   rows=[records[n] for n in ids];frames=np.array([int(r['frames']) for r in rows])
   assert frames.tolist()==[r['frames'] for r in selected]
   summary=dict(complete=True,model=model,checkpoint_epoch=identity['epoch'],samples=len(rows),frames=int(frames.sum()),metrics={})
   for key in metrics:
    values=np.array([float(r[key]) for r in rows]);summary['metrics'][key]=dict(action_mean=float(values.mean()),action_p95=float(np.percentile(values,95)),pooled=float(np.average(values,weights=frames)),observations=int(frames.sum()*29))
   if label=='all':
    original=json.loads((run/'summary.json').read_text())
    for key in metrics:
     for agg in ('action_mean','pooled'):np.testing.assert_allclose(summary['metrics'][key][agg],original['metrics'][key][agg],rtol=1e-6,atol=1e-4)
   write(directory/f'{model}_summary.json',summary)
   with (directory/f'{model}_per_sample.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
   tables.append(dict(filter=label,model=model,epoch=identity['epoch'],samples=len(rows),**{key:summary['metrics'][key]['action_mean'] for key in metrics}))
   summaries[f'{label}/{model}']=summary
 with (OUT/'comparison.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
 count_summary={label:dict(kept=len(rows),excluded=len(test)-len(rows),frames=sum(r['frames'] for r in rows)) for label,rows in datasets.items()}
 manifest=dict(complete=True,method='Reuse deterministic full-test per-clip predictions; apply archived tracker source filter and reaggregate, no retraining or checkpoint reselection',
  test_samples=len(test),counts=count_summary,
  source_hashes={str(p):sha(p) for p in (META,CODE,OLD/'report.json',OLD/'excluded.jsonl',SUPER,DATA/'index.jsonl',DATA/'test.txt')},
  evaluation_hashes={model:sha(run/'per_sample.csv') for model,run in runs.items()},identities=identities,
  mimiclite_policy='Exact historical classifier and metadata hashes; reapply to native windows by full source path, preserving desc_4-only review retention',
  supermimic_policy='Reuse archived keep decisions including feasibility filtering; all source paths and source CSV hashes matched',
  review_retained=sum(r['mimiclite_keep'] and r['mimiclite_review'] for r in audit),
  mimiclite_excluded_but_contact_allowed=sum(not r['mimiclite_keep'] and r['foot_contact_eligible'] for r in audit),
  caveat='Rule screening does not guarantee no environment interaction or physical trackability. RVQ train-source overlap from previous comparison still applies.',
  stairs_example=[r for r in audit if r['sample_id']=='stair_slide_down_R_003__A304_win000000'])
 write(OUT/'manifest.json',manifest);write(OUT/'comparison.json',tables)
 print(json.dumps(dict(counts=count_summary,results=tables,review_retained=manifest['review_retained'],additional_exclusions=manifest['mimiclite_excluded_but_contact_allowed'],stairs=manifest['stairs_example']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
