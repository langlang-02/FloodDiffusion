#!/usr/bin/env python3
import csv,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rot68_reconstruction_comparison'
def main():
 selections=json.loads((OUT/'selection.json').read_text())
 runs={k:[json.loads(l) for l in (ROOT/'outputs'/n/'metrics.jsonl').read_text().splitlines()] for k,n in [('native38','g1_causal_textop_500ep'),('rot6d68','g1_rot6d68_textop_500ep_bs2048')]}
 fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for i,(name,rows) in enumerate(runs.items()):
  steps=[r['step'] for r in rows]
  ax[i,0].plot(steps,[r['val']['rec'] for r in rows],label='Validation rec');ax[i,0].set_yscale('log');ax[i,0].set(title=f'{name}: its own normalized reconstruction loss',xlabel='Optimizer updates',ylabel='Huber loss')
  ax[i,1].plot(steps,[r['lr'] for r in rows],color='#bc6b29');ax[i,1].set(title=f'{name}: learning-rate schedule',xlabel='Optimizer updates',ylabel='Learning rate')
  for a in ax[i]:a.grid(alpha=.2)
 ax[0,0].axvline(79734,color='grey',linestyle='--');ax[0,1].axvline(79734,color='grey',linestyle='--')
 fig.savefig(OUT/'training_curves.png',dpi=180);plt.close(fig)
 labels={'walk':'行走','turn':'转身','dance':'舞蹈','jump':'跳跃','gesture':'挥手组合动作','stairs':'楼梯下滑','drift':'原 native38 的较大平移误差案例'}
 names={'rvq':'RVQ rot6d68','vae_best':'VAE native38 best','rot68':'本次 VAE rot6d68'}
 rows=[]
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>rot6d68 VAE 重建结果</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1350px;margin:auto;padding:25px;background:#f3f6fa;color:#24344a}article,section{padding:24px;margin:20px 0;background:white;border-radius:12px}video,img{width:100%}th,td{padding:8px 20px;text-align:left;border-bottom:1px solid #dde4ec}table{border-collapse:collapse;width:100%}a{color:#1763a9}small{overflow-wrap:anywhere}.note{background:#fff0ce;padding:12px}button,select{font:inherit;padding:5px}</style><h1>本次 rot6d68 VAE：第 500 轮重建</h1><p>四栏从左至右为 GT、RVQ rot6d68、上次 native38 VAE best、本次 rot6d68 VAE best。沿用之前同一组 7 条测试动作，没有按本次误差重新挑选。20 Hz，统一跟随 GT 的相机；保留预测的根平移、朝向与关节误差，无接触修正。</p><p class="note">这些可视化不是严格的架构消融：训练表征、batch 与参数更新次数不同；部分动作源在 RVQ 训练清单中出现过。楼梯几何不可得，只显示参考平面。</p><p>速度：<select id="speed"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select> <button id="pause">暂停全部</button></p>'''
 for r in selections:
  slug=r['slug'];page+=f'<article><h2>{labels[slug]}</h2><video controls loop playsinline preload="metadata" src="videos/{slug}_close.mp4" poster="previews/{slug}_close.jpg"></video><p><a href="videos/{slug}_close.mp4">下载视频</a> · {r["frames"]} 帧</p><small>{html.escape(r["id"])}</small><table><tr><th>模型</th><th>全局 MPJPE (mm)</th><th>去根平移 MPJPE (mm)</th></tr>'
  for k,name in names.items():
   m=r['metrics'][k];page+=f'<tr><td>{name}</td><td>{m["global_mpjpe_mm"]:.2f}</td><td>{m["root_relative_mpjpe_mm"]:.2f}</td></tr>'
   rows.append(dict(category=slug,sample_id=r['id'],model=name,**m))
  page+='</table></article>'
 page+='<section><h2>参数更新次数与学习率</h2><img src="training_curves.png" alt="两次训练各自的loss与学习率曲线"><p>两种表征的 rec 数值不能直接比较优劣；图用于分别查看收敛趋势。native38 虚线是第137轮后 batch 从256增为8192的位置。当前 rot6d68 总计5000次更新，native38总计86631次。</p><p>本次末期验证rec：epoch400=0.046302，450=0.045784，500=0.045617。存在继续优化的可能，但不能用训练loss下降保证测试MPJPE改善。此报告未启动续训。</p><a href="per_clip_metrics.csv">逐动作指标 CSV</a></section><script>const vs=[...document.querySelectorAll("video")];document.querySelector("#speed").onchange=e=>vs.forEach(v=>v.playbackRate=+e.target.value);document.querySelector("#pause").onclick=()=>vs.forEach(v=>v.pause());vs.forEach(v=>v.onplay=()=>vs.forEach(o=>{if(o!==v)o.pause()}));</script></html>'
 (OUT/'index.html').write_text(page)
 with (OUT/'per_clip_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/'training_analysis.json').write_text(json.dumps({k:[{field:r[field] for field in ('epoch','step','lr','train','val')} for r in records] for k,records in runs.items()}))
 print('Report ready:',OUT/'index.html')
if __name__=='__main__':main()
