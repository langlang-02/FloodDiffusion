"""Check four-rank resume, MPJPE selection, and unchanged MoMask parity on smoke data."""
import sys,json,csv
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import G1Kinematics,metric_values,reduce_values
from g1.model import G1CausalVAE
from g1.root_temporal_loss import TEMPORAL_WEIGHTS
from g1.temporal_metrics import temporal_values
from g1.rot6d69 import encode_native,decode_native
from g1.continuation import fixed_rows
out=ROOT/'outputs/g1_rot69_root_temporal_smoke';res=ROOT/'outputs/g1_rot69_root_temporal_resume_smoke'
a=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);b=torch.load(res/'latest.pt',map_location='cpu',weights_only=False)
assert a['identity']['weights']==TEMPORAL_WEIGHTS and len(TEMPORAL_WEIGHTS)==9 and 'dof_acc' not in TEMPORAL_WEIGHTS
assert a['identity']==b['identity'] and a['epoch']==b['epoch']==2 and a['step']==b['step']==2
maximum=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
assert maximum<1e-6,maximum
for p in (out,res):
 assert json.loads((p/'status.json').read_text())['evaluation_complete']
 evals=[json.loads(q.read_text()) for q in (p/'eval_fixed_val').glob('*/summary.json')]
 chosen=min(evals,key=lambda v:v['metrics']['global_mpjpe_mm']['action_mean'])
 best=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
 assert best['epoch']==best['best_mpjpe_epoch']==chosen['epoch']
 assert best['best_mpjpe']==chosen['metrics']['global_mpjpe_mm']['action_mean']
data=ROOT/'outputs/g1_smoke_data_v2';rot=ROOT/'outputs/g1_rot69_smoke_data'
c=torch.load(out/'epoch_0001.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
m=G1CausalVAE(np.load(rot/'mean.npy'),np.load(rot/'std.npy'),**c['identity']['model_params']).cuda().eval();m.load_state_dict(c['model'])
fk=G1Kinematics(data/'kinematic.xml');rows,_=fixed_rows(data,ROOT/'outputs/rot68_fixed_val_smoke')
refs={v['sample_id']:v for v in csv.DictReader((out/'eval_fixed_val/epoch_0001/per_sample.csv').open())};delta=0.
for r in rows:
 g=np.load(data/'motions'/f"{r['id']}.npy");x,anc=encode_native(g)
 with torch.inference_mode():p=decode_native(m.reconstruct(torch.tensor(x[None],device='cuda'),sample=False)[0],torch.tensor(anc[:1],device='cuda',dtype=torch.float32))[0].cpu().numpy()
 vals=metric_values(g,p,fk.positions(g),fk.positions(p),20,fk.endpoints)
 vals.update(temporal_values(g,p,20))
 for k in ('global_mpjpe_mm','root_relative_mpjpe_mm','root_angular_velocity_rmse_rad_s','root_angular_acceleration_rmse_rad_s2'):
  expected=reduce_values(k,vals[k]);actual=float(refs[r['id']][k]);delta=max(delta,abs(expected-actual));np.testing.assert_allclose(expected,actual,atol=.001,rtol=1e-5)
report=dict(four_gpu_resume_model_max_abs=maximum,fixed_val_momask_max_abs_mm=delta,mpjpe_best_selection=True,final_auto_eval=True)
(ROOT/'outputs/rot69_root_temporal_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
