"""Check frozen-cohort MPJPE against unchanged MoMask metrics and saved best selection."""
import json,csv,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import G1Kinematics,metric_values,reduce_values
from g1.model import G1CausalVAE
from g1.rot6d68 import encode_native,decode_native
from g1.continuation import fixed_rows

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
out=ROOT/'outputs/rot68_continuation_smoke';data=ROOT/'outputs/g1_smoke_data_v2';rot=ROOT/'outputs/g1_rot68_smoke_data'
c=torch.load(ROOT/'outputs/g1_rot68_pipeline_smoke/latest.pt',map_location='cpu',weights_only=False)
model=G1CausalVAE(np.load(rot/'mean.npy'),np.load(rot/'std.npy'),**c['identity']['model_params']).cuda().eval();model.load_state_dict(c['model'])
rows,_=fixed_rows(data,ROOT/'outputs/rot68_fixed_val_smoke');fk=G1Kinematics(data/'kinematic.xml')
ref={r['sample_id']:r for r in csv.DictReader((out/'eval_fixed_val/epoch_0002/per_sample.csv').open())};maximum=0.
for r in rows:
 gt=np.load(data/'motions'/f"{r['id']}.npy");feature,anchors=encode_native(gt)
 with torch.inference_mode():pred=decode_native(model.reconstruct(torch.tensor(feature[None],device='cuda'),sample=False)[0],torch.tensor(anchors[:1],device='cuda',dtype=torch.float32))[0].cpu().numpy()
 v=metric_values(gt,pred,fk.positions(gt),fk.positions(pred),20,fk.endpoints)
 for k in ('global_mpjpe_mm','root_relative_mpjpe_mm'):
  expected=reduce_values(k,v[k]);actual=float(ref[r['id']][k]);maximum=max(maximum,abs(expected-actual));np.testing.assert_allclose(expected,actual,rtol=1e-5,atol=.001)
evals=[json.loads(p.read_text()) for p in (out/'eval_fixed_val').glob('*/summary.json')]
assert sorted(x['epoch'] for x in evals)==[2,4,6,8]
best=min(evals,key=lambda v:v['metrics']['global_mpjpe_mm']['action_mean'])
b=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
assert b['epoch']==b['best_mpjpe_epoch']==best['epoch'] and b['best_mpjpe']==best['metrics']['global_mpjpe_mm']['action_mean']
assert json.loads((out/'status.json').read_text())['evaluation_complete']
print(json.dumps(dict(fixed_val_momask_parity=True,max_abs_mm=maximum,periodic_interval_passed=True,best_selection_passed=True,final_auto_eval_passed=True)),flush=True)
