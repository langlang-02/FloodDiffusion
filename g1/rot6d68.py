"""Canonical MoMask local rot6d68 inputs with differentiable native G1 decoding."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from .common import sha256
from .data import read_index
from .kinematics import rot6d
from .model import TextOpLoss,masked_huber,DEFAULT_WEIGHTS
from g1_io import G1_BANK_LAYOUT,model_features_to_feature_bank

FIELDS=['joint_angles','joint_vel','root_roll_pitch_rot6d','root_yaw_vel','root_xy_vel_body','root_height']
INDICES=[i for name in FIELDS for i in range(*G1_BANK_LAYOUT[name])]
SPEC=dict(id='g1_local_rot6d68_v1',fields=FIELDS,feature_dim=68,fps=20,velocity_unit='per_frame',velocity_frame='current_heading')

def encode_native(native):
    bank,anchors=model_features_to_feature_bank(native)
    return bank[:,INDICES].copy(),anchors

def decode_native(feature,anchor):
    """B,T,68 -> B,T,38. Initial GT xy/yaw only; height is predicted."""
    yaw=anchor[:,None,3]+torch.cat((feature.new_zeros(feature.shape[0],1),feature[:,1:,64].cumsum(1)),1)
    c,s=yaw.cos(),yaw.sin()
    dx,dy=feature[...,65],feature[...,66]
    world=torch.stack((c*dx-s*dy,s*dx+c*dy),-1)
    xy=anchor[:,None,:2]+torch.cat((world.new_zeros(world.shape[0],1,2),world[:,1:].cumsum(1)),1)
    zeros=torch.zeros_like(c);ones=torch.ones_like(c)
    heading=torch.stack((c,-s,zeros,s,c,zeros,zeros,zeros,ones),-1).reshape(*yaw.shape,3,3)
    rotation=heading@rot6d(feature[...,58:64])
    six=torch.cat((rotation[...,0],rotation[...,1]),-1)
    return torch.cat((xy,feature[...,67:68],six,feature[...,:29]),-1)

class Rot68Dataset(Dataset):
    def __init__(self,native_root,rot_root,split,window=65):
        self.native_root=Path(native_root);self.rot_root=Path(rot_root)
        self.rows=read_index(native_root,split);self.training=split=='train';self.window=window
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];name=row['id']+'.npy'
        native=np.load(self.native_root/'motions'/name)
        feature=np.load(self.rot_root/'motions'/name)
        contact=np.load(self.native_root/'contacts'/name)
        # Compute deltas before random training crops. Anchor corresponds to crop's first frame.
        start=np.random.randint(len(native)-self.window+1) if self.training and len(native)>self.window else 0
        stop=min(start+self.window,len(native)) if self.training else len(native)
        from .common import extract_root_yaw,rotation_6d_to_matrix
        anchor=np.concatenate((native[start,:3],[extract_root_yaw(rotation_6d_to_matrix(native[start:start+1,3:9]))[0]]))
        return dict(feature=torch.from_numpy(feature[start:stop].copy()),native=torch.from_numpy(native[start:stop].copy()),
                    anchor=torch.tensor(anchor,dtype=torch.float32),contact=torch.from_numpy(contact[start:stop].copy()),
                    eligible=row['foot_contact_eligible'],length=stop-start)

def collate_rot68(items):
    maximum=max(x['length'] for x in items);result={}
    for key in ('feature','native','contact'):
        values=[]
        for x in items:
            v=x[key];pad=maximum-len(v)
            tail=torch.zeros(pad,*v.shape[1:],dtype=v.dtype) if key=='contact' else v[-1:].expand(pad,-1)
            values.append(torch.cat((v,tail)))
        result[key]=torch.stack(values)
    result.update(anchor=torch.stack([x['anchor'] for x in items]),length=torch.tensor([x['length'] for x in items]),eligible=torch.tensor([x['eligible'] for x in items],dtype=torch.bool))
    return result

class Rot68Forward(nn.Module):
    def __init__(self,vae,fk):
        super().__init__();self.vae=vae;self.criterion=TextOpLoss(fk,fps=20)
    def forward(self,batch,sample=True):
        pred,mu,logvar=self.vae.reconstruct(batch['feature'],sample=sample)
        native=decode_native(pred,batch['anchor'])
        # Reuse unchanged physical losses; replace the intermediate native reconstruction term.
        normalizer=SimpleNamespace(mean=native.new_zeros(38),std=native.new_ones(38))
        terms=self.criterion(normalizer,native,batch['native'],mu,logvar,batch['length'],batch['contact'],batch['eligible'])
        valid=torch.arange(pred.shape[1],device=pred.device)[None]<batch['length'][:,None]
        terms['rec']=masked_huber((pred-self.vae.mean)/self.vae.std,(batch['feature']-self.vae.mean)/self.vae.std,valid)
        terms['total']=sum(DEFAULT_WEIGHTS[k]*terms[k] for k in DEFAULT_WEIGHTS)
        return terms

def validate_rot_data(native_root,rot_root):
    m=json.loads((Path(rot_root)/'manifest.json').read_text())
    if not m.get('complete') or m['representation']!=SPEC:raise ValueError('Incomplete or incompatible rot6d68 data')
    if m['native_manifest_sha256']!=sha256(Path(native_root)/'manifest.json'):raise ValueError('Native source changed')
    for n in ('mean','std'):
        if sha256(Path(rot_root)/f'{n}.npy')!=m[f'{n}_sha256']:raise ValueError('Rot68 statistics changed')
    if m['conversion_sha256']!=sha256(Path(__file__).parents[2]/'momask-codes/dataset/bones-seed_prepare/g1_io.py'):raise ValueError('Canonical converter changed')
    return m
