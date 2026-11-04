#!/usr/bin/env python3
"""Fixed staged-prompt inference and faithful MuJoCo G1 replay."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'momask-codes/dataset/bones-seed_prepare'),str(ROOT.parent/'momask-codes/g1_visualize')]
from g1.segment_generation import SegmentDiffusion
from g1.generation import decode_motion
from g1.model import G1CausalVAE
from g1.rot6d69 import decode_native
from g1.common import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--seed',type=int,default=3407);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 c=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=c['identity']['config'];device=torch.device('cuda')
 model=SegmentDiffusion(cfg['clip'],device,cfg['hidden_dim'],cfg['layers'],False).to(device).eval();model.model.load_state_dict(c['ema'])
 texts=['A person waves their right hand.','A person walks forward.','A person squats down.'];ends=[10,25,35]
 torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(2)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  z=model.generate(dict(feature_length=torch.tensor([35],device=device),text=[texts],feature_text_end=[ends]))['generated'][0]
 v=torch.load(ROOT/cfg['vae'],map_location='cpu',weights_only=False)
 assert sha256(ROOT/cfg['vae'])==c['identity']['vae_sha256']
 vae=G1CausalVAE(np.zeros(69,np.float32),np.ones(69,np.float32),**v['identity']['model_params']).to(device).eval();vae.load_state_dict(v['model'])
 with torch.inference_mode():motion=decode_native(decode_motion(vae,z[None]),torch.zeros(1,3,device=device))[0].cpu().numpy()
 assert motion.shape==(137,38) and np.isfinite(motion).all()
 np.save(a.output/'motion.npy',motion)
 manifest=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),epoch=c['epoch'],step=c['step'],smoke=cfg['smoke'],text=texts,token_text_end=ends,seed=a.seed,frames=len(motion),fps=20,weights='EMA',purpose='pipeline smoke only; not trained generation quality' if cfg['smoke'] else 'staged text generation preview')
 del model,vae,c,v;torch.cuda.empty_cache()
 source=ROOT.parent/'MimicLite/active-adaptation/active_adaptation/assets/G1/g1_29dof_nohand/g1_29dof_nohand.xml'
 tree=ET.parse(source);xml=tree.getroot();xml.find('compiler').set('meshdir',str(source.parent.parent/'mjcf/meshes'))
 ET.SubElement(xml.find('worldbody'),'geom',name='segment_preview_floor',type='plane',size='0 0 .1',rgba='.75 .78 .82 1',contype='0',conaffinity='0')
 ET.SubElement(xml.find('worldbody'),'light',pos='0 -3 6',dir='0 .3 -1',directional='true',diffuse='.8 .8 .8',ambient='.3 .3 .3')
 path=a.output/'scene.xml';tree.write(path)
 import mujoco
 from replay_g1 import qpos_addresses,render_video
 model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);root,joints=qpos_addresses(model)
 from types import SimpleNamespace
 args=SimpleNamespace(output_mp4=a.output/'wave_walk_squat.mp4',width=960,height=576,camera_azimuth=135,camera_elevation=-18,camera_distance=3.4,orbit_degrees=0.,clip_joint_limits=False)
 render_video(model,data,motion,20,20,root,joints,args)
 manifest.update(mjcf=str(source),mjcf_sha256=sha256(source),rendering='kinematic qpos replay; no clipping, smoothing or physics correction')
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
