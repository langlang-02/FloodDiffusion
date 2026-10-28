#!/usr/bin/env python3
"""Build comparison tables, source-overlap audit, plots and an offline video gallery."""
import csv,html,json,re,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1.common import G1Kinematics,sha256
OUT=ROOT/'outputs/vae_rvq_comparison';RUN=ROOT/'outputs/g1_causal_textop_500ep';DATA=ROOT/'outputs/bones_g1_native38'
METRICS=['global_mpjpe_mm','root_relative_mpjpe_mm']
NAMES={'rvq':'RVQ rot6d68 (ep499)','vae_best':'Causal VAE native38 best (ep149)','vae_latest':'Causal VAE native38 latest (ep500)'}
COLORS={'gt':'#4384dc','rvq':'#d99226','vae_best':'#21a683','vae_latest':'#a772db'}
def loadcsv(path):return {r['sample_id']:r for r in csv.DictReader(path.open())}
def main():
 historical=[dict(model=n,global_mpjpe_mm=g,root_relative_mpjpe_mm=r,source='user-provided previous server results; checkpoint and split not supplied') for n,g,r in [
 ('RVQ rot6d68',101.66,28.52),('RVQ sincos66',103.03,29.28),('RVQ motion_ae70',100.93,27.43),('Previous Causal VAE rot6d68',146.53,43.67)]]
 dirs={'rvq':OUT/'rvq_same_test','vae_best':RUN/'eval_best','vae_latest':RUN/'eval_latest'}
 summaries={k:json.loads((d/'summary.json').read_text()) for k,d in dirs.items()}
 rows={k:loadcsv(d/'per_sample.csv') for k,d in dirs.items()}
 assert all(set(v)==set(rows['rvq']) for v in rows.values())
 identities={k:json.loads((d/'identity.json').read_text()) for k,d in dirs.items()}
 for field in ('split_sha256','mjcf_sha256','metric_sha256'):
  assert len({v[field] for v in identities.values()})==1,field
 table=[]
 for k,s in summaries.items():
  for agg in ('action_mean','pooled'):
   table.append(dict(model=NAMES[k],samples=s['samples'],aggregation=agg,**{m:s['metrics'][m][agg] for m in METRICS}))
 tracking=Path('/data/langlang/dataset/mimiclite-rvq-tracking/rot6d68_ep499_v1')
 def source(n):return re.sub('_M$','',Path(n).stem.split('_seg')[0].split('_win')[0])
 old={s:{source(n) for n in (tracking/f'{s}.txt').read_text().splitlines()} for s in ('train','val','test')}
 common=[n for n in rows['rvq'] if source(n) in old['test'] and source(n) not in old['train']|old['val']]
 (OUT/'candidate_common_test_samples.txt').write_text('\n'.join(sorted(common))+'\n')
 subset=[]
 for k,rs in rows.items():
  subset.append(dict(model=NAMES[k],samples=len(common),aggregation='action_mean',**{m:float(np.mean([float(rs[n][m]) for n in common])) for m in METRICS}))
 result=dict(historical=historical,matched_full=table,candidate_common_test=subset,
  caveats=['Historical table is user-provided; original split/checkpoint/aggregation not verified.',
   'Full matched-input evaluation is VAE held-out, but 6815 clip source basenames match RVQ train export.',
   'Candidate shared held-out subset matches source basenames to RVQ exported test list and excludes exported train/val sources; original RVQ split with dates unavailable.',
   'RVQ native conversion recomputes deltas within each independent clip (first deltas zero), unlike a feature bank computed before windowing.',
   'Results do not isolate architecture: representations, losses, latent capacity and training schedules differ.'],
  identities=identities)
 (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
 for name,data in [('matched_metrics',table),('candidate_common_test_metrics',subset),('historical_metrics',historical)]:
  with (OUT/f'{name}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 # Exportable chart of same-input scores.
 fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
 keys=list(NAMES)
 for ax,m,title in zip(axes,METRICS,['Global MPJPE (mm)','Root-translation-relative MPJPE (mm)']):
  values=[summaries[k]['metrics'][m]['action_mean'] for k in keys]
  bars=ax.bar(range(3),values,color=[COLORS[k] for k in keys],width=.6)
  ax.bar_label(bars,fmt='%.2f',padding=4);ax.set_xticks(range(3),['RVQ ep499','VAE best ep149','VAE latest ep500']);ax.set_title(title);ax.set_ylim(0,max(values)*1.2);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Same 8,333 input clips | equal-clip means | split caveat applies')
 fig.savefig(OUT/'mpjpe_comparison.png',dpi=180);plt.close(fig)
 selections=json.loads((OUT/'selection.json').read_text());fk=G1Kinematics(DATA/'kinematic.xml')
 perclip=[]
 for r in selections:
  motions={k:np.load(OUT/'clips'/r['slug']/f'{k}.npy') for k in ('gt',*NAMES)}
  gt=motions['gt'];points=fk.positions(gt);t=np.arange(len(gt))/20
  fig,ax=plt.subplots(1,3,figsize=(14,3.7),layout='constrained')
  for k,x in motions.items():
   ax[0].plot(x[:,0],x[:,1],label='GT' if k=='gt' else NAMES[k],color=COLORS[k]);ax[0].scatter(x[0,0],x[0,1],color=COLORS[k],s=15)
   if k!='gt':
    pred=fk.positions(x)
    global_error=np.linalg.norm(pred-points,axis=-1).mean(1)*1000
    relative=np.linalg.norm((pred-x[:,None,:3])-(points-gt[:,None,:3]),axis=-1).mean(1)*1000
    ax[1].plot(t,global_error,color=COLORS[k]);ax[2].plot(t,relative,color=COLORS[k])
    perclip.append(dict(category=r['slug'],sample_id=r['id'],model=NAMES[k],frames=len(gt),**r['metrics'][k]))
  ax[0].set(title='Root XY path',xlabel='X (m)',ylabel='Y (m)');ax[0].axis('equal');ax[0].legend(fontsize=7)
  for a,title in zip(ax[1:],['Global MPJPE per frame','Root-relative MPJPE per frame']):a.set(title=title,xlabel='Time (s)',ylabel='mm')
  for a in ax:a.grid(alpha=.2)
  fig.suptitle(r['label']);fig.savefig(OUT/'previews'/f"{r['slug']}_errors.png",dpi=150);plt.close(fig)
 with (OUT/'selected_clip_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(perclip[0]));w.writeheader();w.writerows(perclip)
 def table_html(rs):
  return '<table><thead><tr><th>模型</th><th>全局 MPJPE</th><th>去根平移 MPJPE</th></tr></thead><tbody>'+''.join(f"<tr><td>{html.escape(r['model'])}</td><td>{r[METRICS[0]]:.2f} mm</td><td>{r[METRICS[1]]:.2f} mm</td></tr>" for r in rs)+'</tbody></table>'
 current=[x for x in table if x['aggregation']=='action_mean']
 history_all=[dict(r,model=r['model']+'（历史）') for r in historical]+[r for r in current if r['model'].startswith('Causal')]
 labels={'walk':'行走','turn':'转身','dance':'舞蹈','jump':'跳跃','gesture':'挥手组合动作','stairs':'楼梯下滑（地形动作）','drift':'较大平移误差案例'}
 content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 重建对比：RVQ / Causal VAE</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#203047;background:#f4f6f9;max-width:1320px;margin:auto;padding:30px}h1,h2{line-height:1.3}section,article{background:white;padding:24px;margin:24px 0;border-radius:14px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:11px;border-bottom:1px solid #dce3ec}.note{background:#fff4d9;padding:16px;border-radius:8px}video,img{max-width:100%;border-radius:8px}a{color:#1763a9}code{overflow-wrap:anywhere}select,button{font:inherit;padding:6px 12px}summary{cursor:pointer}small{color:#627288}</style><h1>G1 动作重建：RVQ 与 Causal VAE</h1><p>同输入测试共 8,333 条动作 / 1,043,739 帧；20 Hz。MoMask 的 29 个命名关节位置，单位 mm。去根平移指标不去根旋转，不做 Procrustes 对齐。</p>'''
 content+='<section><h2>与之前服务器的结果并列</h2>'+table_html(history_all)+'<p class="note">历史四行由用户提供，尚无原始测试集、权重与汇总文件；本次两行是 native38 VAE 的逐动作平均。因此本表是历史结果并列，不能据此直接判断模型优劣。之前的 Causal VAE rot6d68 与本次 native38 模型不同。</p></section>'
 content+='<section><h2>本机同输入复测</h2>'+table_html(current)+'<p class="note">相同动作、帧数、MJCF 和指标代码。8,333 条动作对 VAE 是测试集，但其中 6,815 条源名称匹配 RVQ 训练导出清单；因此这是同输入重建比较，并非严格共同未见测试集。</p><img src="mpjpe_comparison.png" alt="同输入 MPJPE 对比"></section>'
 content+='<section><h2>候选共同测试源交集：'+str(len(common))+' 条</h2>'+table_html(subset)+'<p>按原动作 basename 匹配 RVQ 导出的 test 清单，并排除其 train / val 源。历史清单缺少源日期，故该交集仍需原始 RVQ split 文件最终核验。</p><p><a href="matched_metrics.csv">完整指标 CSV（含 pooled）</a> · <a href="comparison.json">结果与权重溯源 JSON</a> · <a href="selected_clip_metrics.csv">视频样本指标 CSV</a></p></section>'
 content+='<section><h2>同步四栏动作重建</h2><p>左至右：GT、RVQ、VAE best、VAE latest。四栏共用跟随 GT 的近景相机、保留原始预测平移；没有地面吸附、关节限位裁剪或物理修正。每类选取 VAE best 全局误差接近中位数的非镜像长片段，另加一个步行动作 95 分位误差案例。</p><p>播放速度：<select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="0.25">0.25×</option></select> <button id="pause">暂停全部</button></p></section>'
 for r in selections:
  slug=r['slug'];content+=f'<article><h2>{labels[slug]}</h2><video controls loop playsinline preload="metadata" poster="previews/{slug}_close.jpg" src="videos/{slug}_close.mp4"></video><p><a href="videos/{slug}_close.mp4">近景 MP4</a> · <a href="videos/{slug}.mp4">固定远景 MP4</a> · {r["frames"]} 帧 · {r["frames"]/20:.2f} 秒</p><small><code>{html.escape(r["id"])}</code></small>'
  if slug=='stairs':content+='<p class="note">该动作不参与 foot_contact loss。原始楼梯几何不可得，视频只显示参考平面；不能据此评价与楼梯的接触或动力学稳定性。</p>'
  content+=f'<details><summary>根轨迹与逐帧误差</summary><img src="previews/{slug}_errors.png" alt="根轨迹与误差"></details></article>'
 content+='<section><h2>解释边界与复现</h2><p>RVQ 使用本地冻结 rot6d68 ep499 的全部 6 层量化码；VAE 使用 posterior mean。原生输入转换 RVQ 特征时按独立片段重算差分，首帧差分为零，与先算整段差分后切窗的历史 feature bank 存在边界差异。两种模型的表征、损失、潜变量容量和训练计划不同；结果不是仅改变因果性或架构的消融。</p><p>脚本：compare_rvq_native_test.py → prepare_reconstruction_comparison.py → render_reconstruction_comparison.py → report_reconstruction_comparison.py，位于 FloodDiffusion/scripts。selection.json 保存选样规则、source、帧区间、权重 SHA256 与逐样本指标。render_manifest.json 记录相机和渲染参数。</p></section><script>const vs=[...document.querySelectorAll("video")];document.querySelector("#speed").onchange=e=>vs.forEach(v=>v.playbackRate=+e.target.value);document.querySelector("#pause").onclick=()=>vs.forEach(v=>v.pause());vs.forEach(v=>v.addEventListener("play",()=>vs.forEach(o=>{if(o!==v)o.pause()})));</script></html>'
 (OUT/'index.html').write_text(content)
 print(json.dumps(dict(common_test=subset,output=str(OUT/'index.html')),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
