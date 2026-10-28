#!/usr/bin/env python3
import argparse,csv,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import sha256,write_json
from g1.data import read_index
from train_g1_vae import load_data_identity

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data=load_data_identity(a.data_root);rows=read_index(a.data_root,'val')
 code=ROOT.parent/'momask-codes/dataset/bones-seed_prepare/g1_filter_flat_ground.py'
 metadata=Path('/data/langlang/dataset/supermimic/raw/bones_seed/metadata/seed_metadata_v004.csv')
 historical=Path('/data/langlang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1/flat_ground_v1/report.json')
 report=json.loads(historical.read_text())
 for f in (code,metadata):assert sha256(f)==next(v for k,v in report['sha256'].items() if Path(k).name==f.name)
 spec=importlib.util.spec_from_file_location('oldfilter',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 sources={r['source'] for r in rows};audit={}
 with metadata.open() as f:
  for row in csv.DictReader(f):
   if row['move_g1_path'] not in sources:continue
   fields={k:v for k,v in row.items() if v and ('desc' in k or k=='content_name')};fields['source_motion_name']=row['filename'].split('__')[0]
   evidence=module.classify({k:v for k,v in fields.items() if k!='content_natural_desc_4'})
   review=[] if evidence else module.classify({'content_natural_desc_4':fields.get('content_natural_desc_4','')})
   audit[row['move_g1_path']]=dict(evidence=evidence,review=review)
 assert set(audit)==sources
 kept=[r for r in rows if not audit[r['source']]['evidence']]
 assert kept and {r['source_group'] for r in kept}.isdisjoint({r['source_group'] for r in read_index(a.data_root,'test')+read_index(a.data_root,'train')})
 a.output.mkdir(parents=True,exist_ok=False)
 write_json(a.output/'rows.json',kept);write_json(a.output/'source_audit.json',audit)
 (a.output/'samples.txt').write_text('\n'.join(r['id'] for r in kept)+'\n')
 manifest=dict(complete=True,split='val',selection='original validation only; historical MimicLite flat_ground_v1; sole desc4 hits retained',
  native_manifest_sha256=sha256(a.data_root/'manifest.json'),split_sha256=sha256(a.data_root/'val.txt'),rows_sha256=sha256(a.output/'rows.json'),samples_sha256=sha256(a.output/'samples.txt'),
  filter_sha256=sha256(code),metadata_sha256=sha256(metadata),historical_report_sha256=sha256(historical),
  original_samples=len(rows),samples=len(kept),excluded=len(rows)-len(kept),review_retained=sum(bool(audit[r['source']]['review']) for r in kept),
  primary_metric='global_mpjpe_mm/action_mean',secondary_metric='root_relative_mpjpe_mm/action_mean',smoke=data['smoke'])
 write_json(a.output/'manifest.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
