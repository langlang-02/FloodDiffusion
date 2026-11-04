#!/usr/bin/env python3
"""Render synchronized three-way reconstructions with identical cameras and no alignment."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];MOMASK=ROOT.parent/'momask-codes'
sys.path.insert(0,str(MOMASK));sys.path.insert(0,str(MOMASK/'g1_visualize'))
from replay_g1 import qpos_addresses,frame_to_qpos
from evaluation.g1_metrics import G1Kinematics
OUT=ROOT/'outputs/rot69_vs_rot68_comparison'
COLORS=[(67,132,220),(167,114,219),(33,166,131)]
LABELS=['Ground truth','VAE rot68 | epoch 1000','VAE rot69 + abs yaw | epoch 500']
KEYS=['gt','rot68_final','rot69']
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def main():
 follow='--follow' in sys.argv
 aligned='--root-align' in sys.argv
 suffix=('_close' if follow else '')+('_root_aligned' if aligned else '')
 selections=json.loads((OUT/'selection.json').read_text())
 source=ROOT.parent/'MimicLite/active-adaptation/active_adaptation/assets/G1/g1_29dof_nohand/g1_29dof_nohand.xml'
 tree=ET.parse(source);xml=tree.getroot()
 xml.find('compiler').set('meshdir',str(source.parent.parent/'mjcf/meshes'))
 asset=xml.find('asset');ET.SubElement(asset,'texture',name='grid',type='2d',builtin='checker',rgb1='.88 .90 .93',rgb2='.78 .81 .85',width='512',height='512')
 ET.SubElement(asset,'material',name='floor',texture='grid',texrepeat='2 2',reflectance='0')
 world=xml.find('worldbody');ET.SubElement(world,'geom',name='reference_plane',type='plane',size='0 0 .1',material='floor',contype='0',conaffinity='0')
 ET.SubElement(world,'light',pos='0 -3 6',dir='0 .3 -1',directional='true',diffuse='.8 .8 .8',ambient='.3 .3 .3')
 visual=ET.SubElement(xml,'visual');ET.SubElement(visual,'global',offwidth='480',offheight='448');ET.SubElement(visual,'quality',shadowsize='2048')
 path=OUT/'render_scene.xml';tree.write(path)
 model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);root,joints=qpos_addresses(model)
 renderer=mujoco.Renderer(model,height=448,width=480)
 option=mujoco.MjvOption();camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
 camera.azimuth=135;camera.elevation=-18
 # Scene geometry must preserve the exact metric skeleton.
 render_fk=G1Kinematics(path);metric_fk=G1Kinematics(ROOT/'outputs/bones_g1_native38/kinematic.xml')
 videos=OUT/'videos';videos.mkdir(exist_ok=True);previews=OUT/'previews';previews.mkdir(exist_ok=True)
 records=[]
 try:
  for selected in selections:
   motions=[np.load(OUT/'clips'/selected['slug']/f'{k}.npy') for k in KEYS];gt=motions[0];n=len(gt)
   assert all(x.shape==gt.shape for x in motions)
   if aligned:
    motions=[x.copy() for x in motions]
    for x in motions[1:]:x[:,:3]=gt[:,:3]
   np.testing.assert_allclose(render_fk.positions(gt),metric_fk.positions(gt),atol=1e-6,rtol=0)
   # One fixed camera per clip, shared across all models and all frames.
   xyz=np.concatenate([render_fk.positions(x).reshape(-1,3) for x in motions])
   lo,hi=xyz.min(0),xyz.max(0);center=(lo+hi)/2
   camera.lookat[:]=center;camera.distance=max(3.2,float(np.linalg.norm(hi-lo))*1.45)
   if follow:
    relative=np.concatenate([render_fk.positions(x)-gt[:,None,:3] for x in motions],axis=1)
    span=np.max(np.linalg.norm(relative.max(1)-relative.min(1),axis=-1))
    camera.distance=max(3.2,float(span)*1.45)
   movie=videos/f"{selected['slug']}{suffix}.mp4"
   snapshot=None
   with imageio.get_writer(movie,fps=20,codec='libx264',quality=8,macro_block_size=16) as writer:
    for t in range(n):
     if follow:camera.lookat[:]=gt[t,:3]+np.array([0,0,-.15])
     canvas=Image.new('RGB',(1440,576),(19,27,40));draw=ImageDraw.Draw(canvas)
     draw.text((18,10),selected['label'],font=font(24),fill='white')
     draw.text((1030,14),f'{t+1:03d}/{n:03d} frames  |  {t/20:.2f}s / {(n-1)/20:.2f}s',font=font(17),fill='#c3ccd9')
     for i,(key,motion,color,label) in enumerate(zip(KEYS,motions,COLORS,LABELS)):
      # Preserve original material contrast, tint only the lighter robot surfaces.
      if t==0 and i==0:original=model.geom_rgba.copy()
      model.geom_rgba[:]=original
      mask=(model.geom_bodyid!=0)&(original[:,:3].mean(1)>.35)
      model.geom_rgba[mask,:3]=np.array(color)/255
      frame_to_qpos(model,data,motion[t],root,joints,False,t,20)
      renderer.update_scene(data,camera=camera,scene_option=option)
      canvas.paste(Image.fromarray(renderer.render()),(480*i,112))
      draw.text((480*i+12,48),label,font=font(19),fill=color)
      if key=='gt':text='Reference | shared world coordinates'
      else:
       m=selected['metrics'][key];text=f"MPJPE {m['global_mpjpe_mm']:.1f} mm | root-rel {m['root_relative_mpjpe_mm']:.1f} mm"
      draw.text((480*i+12,78),text,font=font(15),fill='#e0e6ef')
     note='20 Hz | '+('shared camera follows GT' if follow else 'identical fixed camera')+' | no root alignment, joint clipping or physics correction'
     if aligned:note='GT root translation applied to all predictions; rotations/joints unchanged | numbers above are ORIGINAL errors'
     if selected['slug']=='stairs' and not aligned:note='Terrain motion: reference plane only; original stairs are unavailable. No contact correction.'
     draw.text((14,560),note,font=font(12),fill='#bdc7d5')
     writer.append_data(np.asarray(canvas))
     if t==n//2:snapshot=canvas.copy()
   snapshot.save(previews/f"{selected['slug']}{suffix}.jpg",quality=93)
   records.append(dict(slug=selected['slug'],frames=n,fps=20,video=str(movie),camera_lookat=camera.lookat.tolist(),camera_lookat_frame=n-1,camera_distance=camera.distance,shared_camera_follows_gt=follow,metric_fk_verified=True))
   print(json.dumps(records[-1]),flush=True)
 finally:renderer.close()
 (OUT/f'render_manifest{suffix}.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
