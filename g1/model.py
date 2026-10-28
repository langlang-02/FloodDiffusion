"""FloodDiffusion Wan 1D causal VAE with the seven active TextOp loss terms."""
import torch
from torch import nn
from torch.nn import functional as F
from models.vae_wan_1d import VAEWanModel
from .kinematics import matrix_quaternion

DEFAULT_WEIGHTS = dict(rec=1., kl=1e-4, body_trans=.05, body_rot=.01,
                       dof_pos=.03, dof_vel=1e-5, foot_contact=.01)


def masked_huber(pred, target, valid):
    error = F.huber_loss(pred, target, reduction='none', delta=1.)
    mask = valid
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(error)
    return (error * mask).sum() / mask.sum().clamp_min(1)


class G1CausalVAE(VAEWanModel):
    def __init__(self, mean, std, **params):
        super().__init__(input_dim=len(mean), **params)
        if self.downsample_factor != 4:
            raise ValueError('Upstream streaming implementation requires downsample factor 4')
        self.mean.copy_(torch.as_tensor(mean))
        self.std.copy_(torch.as_tensor(std))

    def reconstruct(self, features, sample=False):
        length = features.shape[1]
        normalized = (features - self.mean) / self.std
        padding = (-(length - 1)) % 4
        if padding:
            normalized = torch.cat((normalized, normalized[:, -1:].expand(-1, padding, -1)), 1)
        mu, logvar = self.model.encode(normalized.transpose(1, 2), [0, 1], return_dist=True)
        logvar = logvar.clamp(-10, 10)
        z = self.model.reparameterize(mu, logvar) if sample else mu
        pred = self.model.decode(z, [0, 1]).transpose(1, 2)[:, :length]
        if pred.shape != features.shape:
            raise RuntimeError(f'Unexpected output shape: {pred.shape} vs {features.shape}')
        return pred * self.std + self.mean, mu, logvar


class TextOpLoss(nn.Module):
    def __init__(self, fk, weights=None, fps=20.):
        super().__init__()
        self.fk = fk
        self.weights = DEFAULT_WEIGHTS if weights is None else weights
        if set(self.weights) != set(DEFAULT_WEIGHTS):
            raise ValueError('Expected exactly the seven active TextOp v3 loss weights')
        self.fps = fps

    def forward(self, model, pred, gt, mu, logvar, lengths, contact, eligible):
        valid = torch.arange(gt.shape[1], device=gt.device)[None] < lengths[:, None]
        pfk = self.fk(pred)
        with torch.no_grad():
            gfk = self.fk(gt)
        pq, gq = matrix_quaternion(pfk['body_rot']), matrix_quaternion(gfk['body_rot'])
        # Equivalent quaternion signs must not introduce a fictitious rotation error.
        pq = torch.where((pq*gq).sum(-1, keepdim=True) < 0, -pq, pq)
        terms = dict(rec=masked_huber((pred-model.mean)/model.std, (gt-model.mean)/model.std, valid),
                     body_trans=masked_huber(pfk['body_pos'], gfk['body_pos'], valid),
                     body_rot=masked_huber(pq, gq, valid),
                     dof_pos=masked_huber(pred[..., 9:], gt[..., 9:], valid))
        terms['dof_vel'] = masked_huber(torch.diff(pred[..., 9:], dim=1)*self.fps,
                                       torch.diff(gt[..., 9:], dim=1)*self.fps,
                                       valid[:, 1:] & valid[:, :-1])
        latent_valid = torch.arange(mu.shape[-1], device=gt.device)[None] < (1 + (lengths-1+3)//4)[:, None]
        kl = .5 * (mu.square() + logvar.exp() - 1 - logvar)
        terms['kl'] = (kl*latent_valid[:, None]).sum() / (latent_valid.sum()*mu.shape[1]).clamp_min(1)
        feet = self.fk.feet
        gate = contact[..., None].to(gt)
        # All feet/coordinates within eligible valid frames form the denominator,
        # matching TextOp's masked-position Huber; excluded clips do not dilute it.
        terms['foot_contact'] = masked_huber(pfk['anchors'][..., feet, :]*gate,
                                            gfk['anchors'][..., feet, :]*gate,
                                            valid & eligible[:, None])
        terms['total'] = sum(self.weights[key] * value for key, value in terms.items())
        return terms
