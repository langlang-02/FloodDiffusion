"""Exact global batches and denominator-aware DDP reduction for TextOp losses."""
import math
import torch
from torch import nn
from torch.utils.data import Sampler
from .model import DEFAULT_WEIGHTS

KEYS = tuple(DEFAULT_WEIGHTS)


class GlobalBatchShard(Sampler):
    """Partition each global batch; no padding, duplicate, or dropped examples."""
    def __init__(self, size, batch_size, rank, world, seed=3407, training=True):
        if size < world and training:
            raise ValueError('Training dataset must have at least one example per rank')
        if batch_size < world and training:
            raise ValueError('Global training batch must be at least world size')
        self.size, self.batch_size, self.rank, self.world = size, batch_size, rank, world
        self.seed, self.training, self.epoch = seed, training, 0

    def batches(self):
        if self.training:
            indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        else:
            indices = list(range(self.size))
        batches = [indices[i:i+self.batch_size] for i in range(0,self.size,self.batch_size)]
        if self.training and len(batches)>1 and len(batches[-1])<self.world:
            batches[-2].extend(batches.pop())
        return batches

    def __iter__(self):
        for i,batch in enumerate(self.batches()):
            if self.training:
                yield batch[self.rank::self.world]
            elif i % self.world == self.rank:
                # Preserve single-GPU validation batches and frame-weighted scores.
                yield batch

    def __len__(self):
        n = math.ceil(self.size/self.batch_size)
        if self.training:
            return n-int(n>1 and 0<self.size%self.batch_size<self.world)
        return len(range(self.rank,n,self.world))


class LossForward(nn.Module):
    def __init__(self, vae, criterion):
        super().__init__()
        self.vae, self.criterion = vae, criterion

    def forward(self, batch, sample=True):
        pred,mu,logvar = self.vae.reconstruct(batch['feature'],sample=sample)
        return self.criterion(self.vae,pred,batch['feature'],mu,logvar,
                              batch['length'],batch['contact'],batch['eligible'])


def denominators(batch):
    # Constant per-frame channel/joint counts cancel between local/global ratios.
    length = batch['length']
    frames = length.sum().float()
    values = dict(rec=frames,body_trans=frames,body_rot=frames,dof_pos=frames,
                  dof_vel=(length-1).clamp_min(0).sum().float(),
                  kl=(1+(length-1+3)//4).sum().float(),
                  foot_contact=(length*batch['eligible']).sum().float())
    return torch.stack([values[k] for k in KEYS])


def ddp_objective(terms, local_counts, global_counts, world, weights):
    # DDP averages rank gradients: multiply by world to recover global means.
    return sum(weights[k]*terms[k]*world*local_counts[i]/global_counts[i].clamp_min(1)
               for i,k in enumerate(KEYS))
