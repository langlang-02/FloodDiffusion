"""GT-matched temporal derivatives for rot69; no zero-motion smoothing target."""
from types import SimpleNamespace
import torch
from torch import nn
from .model import TextOpLoss,masked_huber,DEFAULT_WEIGHTS
from .kinematics import rot6d,matrix_quaternion
from .rot6d69 import decode_native

TEMPORAL_WEIGHTS={**DEFAULT_WEIGHTS,'body_rot':.05,'dof_vel':.01,
                  'root_rot_vel':.01,'root_rot_acc':1e-4}

def angular_velocity(rotation,fps=20.):
    """World-frame SO(3) log of R[t+1] R[t]^T, in rad/s."""
    delta=rotation[:,1:]@rotation[:,:-1].transpose(-1,-2)
    q=matrix_quaternion(delta)
    q=torch.where(q[...,:1]<0,-q,q)
    v=q[...,1:];n=v.norm(dim=-1,keepdim=True)
    scale=torch.where(n>1e-6,2*torch.atan2(n,q[...,:1])/n.clamp_min(1e-6),torch.full_like(n,2.))
    return v*scale*fps

def temporal_terms(pred,gt,lengths,fps=20.):
    valid=torch.arange(pred.shape[1],device=pred.device)[None]<lengths[:,None]
    pairs=valid[:,1:]&valid[:,:-1]
    triples=valid[:,2:]&valid[:,1:-1]&valid[:,:-2]
    pv=angular_velocity(rot6d(pred[...,3:9]),fps)
    with torch.no_grad():gv=angular_velocity(rot6d(gt[...,3:9]),fps)
    return dict(root_rot_vel=masked_huber(pv,gv,pairs),
                root_rot_acc=masked_huber(torch.diff(pv,dim=1)*fps,torch.diff(gv,dim=1)*fps,triples))

class TemporalForward(nn.Module):
    def __init__(self,vae,fk):
        super().__init__();self.vae=vae;self.criterion=TextOpLoss(fk,fps=20)
    def forward(self,batch,sample=True):
        pred,mu,logvar=self.vae.reconstruct(batch['feature'],sample=sample)
        native=decode_native(pred,batch['anchor'])
        normalizer=SimpleNamespace(mean=native.new_zeros(38),std=native.new_ones(38))
        terms=self.criterion(normalizer,native,batch['native'],mu,logvar,batch['length'],batch['contact'],batch['eligible'])
        valid=torch.arange(pred.shape[1],device=pred.device)[None]<batch['length'][:,None]
        terms['rec']=masked_huber((pred-self.vae.mean)/self.vae.std,(batch['feature']-self.vae.mean)/self.vae.std,valid)
        terms.update(temporal_terms(native,batch['native'],batch['length']))
        terms['total']=sum(TEMPORAL_WEIGHTS[k]*terms[k] for k in TEMPORAL_WEIGHTS)
        return terms
