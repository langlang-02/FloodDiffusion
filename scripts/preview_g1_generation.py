#!/usr/bin/env python3
"""Generate fixed-seed text motions and render the simulation G1 meshes in MuJoCo."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,sys,html,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'momask-codes/dataset/bones-seed_prepare'),str(ROOT.parent/'momask-codes/g1_visualize')]
from g1.generation import G1Diffusion,decode_motion
from g1.model import G1CausalVAE
from g1.rot6d69 import decode_native
from g1.common import sha256
PROMPTS=[('walk','Walk forward','A person walks forward at a normal pace.'),
 ('run','Run forward','A person runs forward.'),
 ('wave','Wave right hand','A person stands and waves their right hand.'),
 ('squat','Squat and stand','A person squats down and stands back up.'),
 ('punch','Boxing punches','A person throws alternating punches with both hands.'),
 ('dance','Dance','A person dances, moving their arms and stepping side to side.')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=161);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'motions').mkdir();(out/'videos').mkdir();(out/'previews').mkdir()
 # Hold a stable checkpoint snapshot even if latest.pt is atomically replaced by training.
 snapshot=out/'checkpoint_snapshot.pt';os.link(Path(a.checkpoint).resolve(),snapshot)
 c=torch.load(snapshot,map_location='cpu',weights_only=False);identity=c['identity'];cfg=identity['config']
 device=torch.device('cuda');torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=G1Diffusion(cfg['clip'],device,cfg['hidden_dim'],cfg['layers']).to(device).eval()
 model.model.load_state_dict(c['ema'],strict=True)
 epoch,step=c['epoch'],c['step'];del c
 vc=torch.load(ROOT/cfg['vae'],map_location='cpu',weights_only=False)
 assert sha256(ROOT/cfg['vae'])==identity['vae_sha256']
 vae=G1CausalVAE(np.zeros(69,np.float32),np.ones(69,np.float32),**vc['identity']['model_params']).to(device).eval().requires_grad_(False)
 vae.load_state_dict(vc['model']);del vc
 records=[];latent_length=1+(a.frames-1+3)//4
 with torch.inference_mode():
  for i,(slug,label,prompt) in enumerate(PROMPTS):
   record=dict(slug=slug,label=label,prompt=prompt,samples=[])
   for seed in (3407,3408):
    torch.manual_seed(seed);np.random.seed(seed)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     z=model.generate(dict(feature_length=torch.tensor([latent_length],device=device),text=[prompt]))['generated'][0]
    features=decode_motion(vae,z[None])[:,:a.frames]
    native=decode_native(features,torch.zeros(1,3,device=device))[0].cpu().numpy()
    assert native.shape==(a.frames,38) and np.isfinite(native).all()
    name=f'{slug}_seed{seed}';np.save(out/'motions'/f'{name}.npy',native)
    np.save(out/'motions'/f'{name}_rot69.npy',features[0].cpu().numpy());np.save(out/'motions'/f'{name}_latent.npy',z.cpu().numpy())
    record['samples'].append(dict(seed=seed,motion=f'motions/{name}.npy',root_height_min=float(native[:,2].min()),root_height_max=float(native[:,2].max()),root_travel_m=float(np.linalg.norm(native[-1,:2]-native[0,:2]))))
   records.append(record);print('Generated '+slug,flush=True)
 manifest=dict(checkpoint=str(Path(a.checkpoint).resolve()),snapshot_sha256=sha256(snapshot),epoch=epoch,step=step,weights='EMA',checkpoint_selection='latest saved; no validation-selected best exists',vae=cfg['vae'],vae_sha256=identity['vae_sha256'],clip_sha256=identity['clip_sha256'],fps=20,frames=a.frames,cfg_scale=model.cfg_scale,noise_steps=model.noise_steps,seeds=[3407,3408],samples=records,rendering='G1 MuJoCo mesh kinematic qpos replay, no physics stepping, joint clipping, smoothing or contact correction',code_sha256=sha256(__file__))
 del model,vae;torch.cuda.empty_cache()
 render(out,manifest)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 page='''<!doctype html><meta charset="utf-8"><title>G1 text-to-motion preview</title><style>body{max-width:1100px;margin:32px auto;background:#101827;color:#e6edf6;font:17px system-ui;padding:20px}video{width:100%;border-radius:8px}section{margin:30px 0;padding:18px;background:#1b283c}a{color:#92c5ff}</style>'''
 page+=f'<h1>G1 文本生成动作 · epoch {epoch} EMA</h1><p>当前最新 checkpoint；未进行验证集最优选择。每条文本使用固定 seed 3407 / 3408，各 {a.frames/20:.2f} 秒，未按效果挑选。</p><p>使用仿真 G1 网格，在 MuJoCo 中逐帧回放；没有动力学跟踪、关节裁剪、平滑或接触修正。相机跟随根位置，动作世界坐标保持原样。</p><h2>六种动作总览（seed 3407）</h2><video controls loop src="videos/overview.mp4" poster="previews/contact_sheet.jpg"></video>'
 for r in records:
  page+=f'<section><h2>{html.escape(r["label"])}</h2><p>{html.escape(r["prompt"])}</p><video controls loop preload="metadata" src="videos/{r["slug"]}.mp4" poster="previews/{r["slug"]}.jpg"></video></section>'
 page+='<p><a href="manifest.json">生成参数与可复现记录</a></p>'
 (out/'index.html').write_text(page)
 print(json.dumps(dict(output=str(out),epoch=epoch,step=step,complete=True)),flush=True)

def render(out,manifest):
 import mujoco,imageio.v2 as imageio
 from PIL import Image,ImageDraw,ImageFont
 from replay_g1 import qpos_addresses,frame_to_qpos
 source=ROOT.parent/'MimicLite/active-adaptation/active_adaptation/assets/G1/g1_29dof_nohand/g1_29dof_nohand.xml'
 tree=ET.parse(source);xml=tree.getroot();xml.find('compiler').set('meshdir',str(source.parent.parent/'mjcf/meshes'))
 asset=xml.find('asset');ET.SubElement(asset,'texture',name='preview_grid',type='2d',builtin='checker',rgb1='.88 .90 .93',rgb2='.78 .81 .85',width='512',height='512')
 ET.SubElement(asset,'material',name='preview_floor',texture='preview_grid',texrepeat='2 2',reflectance='0')
 world=xml.find('worldbody');ET.SubElement(world,'geom',name='preview_plane',type='plane',size='0 0 .1',material='preview_floor',contype='0',conaffinity='0')
 ET.SubElement(world,'light',pos='0 -3 6',dir='0 .3 -1',directional='true',diffuse='.8 .8 .8',ambient='.3 .3 .3')
 visual=xml.find('visual')
 if visual is None:visual=ET.SubElement(xml,'visual')
 global_=visual.find('global')
 if global_ is None:global_=ET.SubElement(visual,'global')
 global_.set('offwidth','480');global_.set('offheight','448')
 scene=out/'render_scene.xml';tree.write(scene)
 model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);root,joints=qpos_addresses(model)
 renderer=mujoco.Renderer(model,height=448,width=480);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
 camera.azimuth=135;camera.elevation=-18;camera.distance=3.4
 fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=lambda n:ImageFont.truetype(fontpath,n)
 overview=[Image.new('RGB',(1440,1024),(19,27,40)) for _ in range(manifest['frames'])]
 try:
  for i,r in enumerate(manifest['samples']):
   motions=[np.load(out/s['motion']) for s in r['samples']]
   with imageio.get_writer(out/'videos'/f'{r["slug"]}.mp4',fps=20,codec='libx264',quality=8,macro_block_size=16) as writer:
    for t in range(manifest['frames']):
     canvas=Image.new('RGB',(960,576),(19,27,40));draw=ImageDraw.Draw(canvas)
     draw.text((18,8),f'{r["label"]} | epoch {manifest["epoch"]} EMA',font=font(23),fill='white')
     draw.text((18,40),r['prompt'],font=font(16),fill='#cad6e5')
     for j,motion in enumerate(motions):
      frame_to_qpos(model,data,motion[t],root,joints,False,t,20)
      camera.lookat[:]=motion[t,:3]+np.array([0,0,-.1])
      renderer.update_scene(data,camera=camera)
      frame=Image.fromarray(renderer.render());canvas.paste(frame,(j*480,96))
      draw.text((j*480+18,70),f'seed {r["samples"][j]["seed"]}',font=font(18),fill='#88d5bd')
      if j==0:
       x,y=(i%3)*480,(i//3)*512
       overview[t].paste(frame,(x,y+56));d=ImageDraw.Draw(overview[t]);d.text((x+12,y+8),r['label'],font=font(22),fill='white')
       d.text((x+12,y+34),f'epoch {manifest["epoch"]} EMA | seed 3407',font=font(14),fill='#cad6e5')
     draw.text((15,552),f'{t/20:.2f}s | MuJoCo G1 mesh | kinematic replay, no physics correction',font=font(15),fill='#cad6e5')
     writer.append_data(np.asarray(canvas))
     if t==manifest['frames']//2:canvas.save(out/'previews'/f'{r["slug"]}.jpg',quality=93)
   print('Rendered '+r['slug'],flush=True)
 finally:renderer.close()
 with imageio.get_writer(out/'videos'/'overview.mp4',fps=20,codec='libx264',quality=8,macro_block_size=16) as writer:
  for frame in overview:writer.append_data(np.asarray(frame))
 overview[manifest['frames']//2].save(out/'previews'/'contact_sheet.jpg',quality=93)
 manifest.update(mjcf=str(source),mjcf_sha256=sha256(source),render_scene_sha256=sha256(scene))

if __name__=='__main__':main()
