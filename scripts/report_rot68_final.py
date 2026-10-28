import json,html
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'outputs/rot68_epoch1000_analysis'
rows=json.loads((out/'selection.json').read_text());diag=json.loads((out/'diagnostics.json').read_text())
s='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>G1 动作重建：epoch 1000</title><style>body{margin:32px auto;max-width:1500px;background:#101827;color:#e6edf6;font:16px system-ui;padding:0 20px}video{width:100%;background:#000}section{margin:28px 0;padding:18px;background:#1d293d;border-radius:12px}table{border-collapse:collapse}th,td{padding:10px 18px;border:1px solid #465366}button{padding:10px;margin:10px;color:white;background:#285b9e;border:0;border-radius:6px}small{color:#bac7dc}a{color:#84b8ff}</style><h1>G1 动作重建：最佳 epoch 1000</h1><p>从左到右：真值、native38 epoch149、rot68 epoch500、rot68 epoch1000。同一动作、同步帧、相同相机。沿用上次选定的 7 个片段，未按新模型效果重新挑选。</p><p>世界坐标模式保留全部原始误差；去根平移模式仅将预测根位置替换为真值，保留预测旋转和关节角度。后者用于观察姿态误差，不是原始重建结果。视频上方数字始终为原始重建的指标。</p><button onclick="mode(false)">世界坐标</button><button onclick="mode(true)">去根平移</button><span id="mode">当前：世界坐标</span><h2>同一完整测试集：8333 个动作</h2><table><tr><th>模型</th><th>去根平移 MPJPE</th><th>额外修正 yaw</th><th>额外修正根旋转</th></tr>'''
for key,label in [('native38','native38 ep149'),('rot68_final','rot68 ep1000')]:
 m=diag[key]['metrics'];s+='<tr><td>'+label+'</td>'+''.join(f'<td>{m[k]:.2f} mm</td>' for k in ('root_relative_mpjpe_mm','gt_yaw_oracle_mm','gt_root_rotation_oracle_mm'))+'</tr>'
s+='</table><p>后两列使用真值进行诊断，不是可部署模型成绩；各误差项不可直接相加。完整测试集包含环境交互动作。</p>'
for r in rows:
 slug=r['slug'];s+=f'<section><h2>{html.escape(r["label"])}</h2><small>{html.escape(r["id"])}</small>'
 if slug=='stairs':s+='<p>地形依赖动作，仅作诊断展示；场景未包含原始楼梯。</p>'
 s+=f'<video controls preload="none" poster="previews/{slug}_close.jpg" data-slug="{slug}" src="videos/{slug}_close.mp4"></video></section>'
s+='''<script>function mode(aligned){document.querySelectorAll('video').forEach(v=>{let t=v.currentTime;v.pause();v.src='videos/'+v.dataset.slug+'_close'+(aligned?'_root_aligned':'')+'.mp4';v.poster='previews/'+v.dataset.slug+'_close'+(aligned?'_root_aligned':'')+'.jpg';v.addEventListener('loadedmetadata',()=>v.currentTime=Math.min(t,v.duration),{once:true});});document.getElementById('mode').textContent=aligned?'当前：去根平移（诊断）':'当前：世界坐标';}</script></html>'''
(out/'index.html').write_text(s)
print(out/'index.html')
