#!/usr/bin/env python3
"""Compare fixed historical clips: GT, rot68 ep1000 and absolute-yaw rot69 ep500."""
import csv,json,sys,shutil,html
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import G1Kinematics,metric_values,reduce_values,sha256
from g1.rot6d69 import encode_native,decode_native,validate_rot_data
from g1.model import G1CausalVAE

def main():
 old=ROOT/'outputs/rot68_epoch1000_analysis';out=ROOT/'outputs/rot69_vs_rot68_comparison';data=ROOT/'outputs/bones_g1_native38';rot=ROOT/'outputs/bones_g1_rot6d69'
 runs=dict(rot68_final=ROOT/'outputs/g1_rot68_continue_ep0500_1000_bs2048_mpjpe',rot69=ROOT/'outputs/g1_rot69_absyaw_500ep_bs2048_mpjpe')
 validate_rot_data(data,rot);out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 c=torch.load(runs['rot69']/'best.pt',map_location='cpu',weights_only=False);assert c['epoch']==500
 m=G1CausalVAE(np.load(rot/'mean.npy'),np.load(rot/'std.npy'),**c['identity']['model_params']).cuda().eval();m.load_state_dict(c['model'])
 refs={k:{r['sample_id']:r for r in csv.DictReader((v/'eval_best/per_sample.csv').open())} for k,v in runs.items()}
 keys=('global_mpjpe_mm','root_relative_mpjpe_mm','root_yaw_deg','joint_mae_deg')
 selections=json.loads((old/'selection.json').read_text());fk=G1Kinematics(data/'kinematic.xml');records=[]
 for r in selections:
  d=out/'clips'/r['slug'];d.mkdir(parents=True)
  for k in ('gt','rot68_final'):shutil.copy2(old/'clips'/r['slug']/f'{k}.npy',d/f'{k}.npy')
  g=np.load(d/'gt.npy');np.testing.assert_array_equal(g,np.load(data/'motions'/f"{r['id']}.npy"));x,anc=encode_native(g);np.testing.assert_array_equal(x,np.load(rot/'motions'/f"{r['id']}.npy"))
  with torch.inference_mode():p=decode_native(m.reconstruct(torch.tensor(x[None],device='cuda'),sample=False)[0],torch.tensor(anc[:1],device='cuda',dtype=torch.float32))[0].cpu().numpy()
  np.save(d/'rot69.npy',p);metrics={}
  for k in runs:
   pred=np.load(d/f'{k}.npy');vals=metric_values(g,pred,fk.positions(g),fk.positions(pred),20,fk.endpoints)
   metrics[k]={q:reduce_values(q,vals[q]) for q in keys}
   for q,v in metrics[k].items():np.testing.assert_allclose(v,float(refs[k][r['id']][q]),atol=.01,rtol=1e-4)
  r['metrics']=metrics;records.append(dict(slug=r['slug'],id=r['id'],metrics=metrics));print(json.dumps(records[-1]),flush=True)
 (out/'selection.json').write_text(json.dumps(selections,indent=2))
 identities={k:dict(checkpoint=str(v/'best.pt'),sha256=sha256(v/'best.pt'),evaluation=json.loads((v/'eval_best/summary.json').read_text())) for k,v in runs.items()}
 (out/'identity.json').write_text(json.dumps(dict(models=identities,selection_source=str(old/'selection.json'),selection_sha256=sha256(old/'selection.json'),all_selected_metrics_verified=True,posterior='mean',fps=20),indent=2))
 s='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>rot69 ep500 vs rot68 ep1000</title><style>body{max-width:1440px;margin:28px auto;padding:0 20px;background:#101827;color:#e6edf6;font:16px system-ui}section{margin:28px 0;padding:18px;background:#1d293d;border-radius:12px}video{width:100%}table{border-collapse:collapse}th,td{padding:10px 16px;border:1px solid #465366}button{padding:12px;margin:10px 10px 10px 0;background:#285b9e;color:white;border:0;border-radius:6px}small{color:#bac7dc}a{color:#84b8ff}</style><h1>绝对 yaw 对重建效果的影响</h1><p>从左到右：真值（蓝）、rot68 epoch1000（紫）、rot69＋绝对 yaw epoch500（绿）。沿用之前固定的 7 个动作片段；没有根据新模型效果重新挑选。视频内三栏同步播放，使用同一相机。</p><p>世界坐标模式保留原始轨迹；去根平移模式把预测根位置替换成真值，仅用于比较姿态，保留预测朝向和关节角度。视频中的指标始终是原始重建成绩。</p><button onclick="mode(false)">世界坐标</button><button onclick="mode(true)">去根平移（诊断）</button><span id="mode">当前：世界坐标</span><h2>完整测试集：8333 个动作</h2><table><tr><th>模型</th><th>全局 MPJPE</th><th>去根平移 MPJPE</th><th>yaw 误差</th></tr>'''
 for k,label in [('rot68_final','rot68 ep1000'),('rot69','rot69 ep500')]:
  v=identities[k]['evaluation']['metrics'];s+=f'<tr><td>{label}</td><td>{v[keys[0]]["action_mean"]:.2f} mm</td><td>{v[keys[1]]["action_mean"]:.2f} mm</td><td>{v[keys[2]]["action_mean"]:.2f}°</td></tr>'
 s+='</table><p>同一完整测试集、动作等权平均，包含地形依赖动作。两次训练的 batch、更新次数和学习率调度不同。</p>'
 for r in selections:
  slug=r['slug'];s+=f'<section><h2>{html.escape(r["label"])}</h2><small>{html.escape(r["id"])}</small>'
  if slug=='stairs':s+='<p>地形依赖动作：原始楼梯不可用，只显示参考地面。</p>'
  s+='<table><tr><th>模型</th><th>全局 MPJPE</th><th>去根平移 MPJPE</th><th>yaw 误差</th></tr>'
  for k,label in [('rot68_final','rot68 ep1000'),('rot69','rot69 ep500')]:
   v=r['metrics'][k];s+=f'<tr><td>{label}</td><td>{v[keys[0]]:.2f} mm</td><td>{v[keys[1]]:.2f} mm</td><td>{v[keys[2]]:.2f}°</td></tr>'
  s+=f'</table><video controls playsinline preload="none" poster="previews/{slug}_close.jpg" data-slug="{slug}" src="videos/{slug}_close.mp4"></video></section>'
 s+='''<script>function mode(a){document.querySelectorAll('video').forEach(v=>{const t=v.currentTime;v.pause();const p=v.dataset.slug+'_close'+(a?'_root_aligned':'');v.addEventListener('loadedmetadata',()=>v.currentTime=Math.min(t,v.duration),{once:true});v.src='videos/'+p+'.mp4';v.poster='previews/'+p+'.jpg';});document.getElementById('mode').textContent=a?'当前：去根平移（诊断）':'当前：世界坐标';}</script></html>'''
 (out/'index.html').write_text(s)
if __name__=='__main__':main()
