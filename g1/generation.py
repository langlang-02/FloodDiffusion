"""G1 adapter for upstream Wan diffusion forcing, with frozen CLIP conditioning."""
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.diffusion_forcing_wan import DiffForcingWanModel
from models.tools import wan_model


def masked_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                     softmax_scale=None, q_scale=None, causal=False, **kwargs):
    # SDPA fallback with actual key lengths; upstream fallback ignores padding.
    mask = torch.ones(q.shape[0], 1, q.shape[1], k.shape[1], device=q.device, dtype=torch.bool)
    if k_lens is not None:
        mask &= torch.arange(k.shape[1], device=q.device)[None,None,None,:] < k_lens.to(q.device)[:,None,None,None]
    if causal:
        mask &= torch.ones(q.shape[1],k.shape[1],device=q.device,dtype=torch.bool).tril()
    if q_scale is not None: q = q*q_scale
    result = F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                           attn_mask=mask,dropout_p=dropout_p,scale=softmax_scale)
    return result.transpose(1,2).contiguous()


class G1Diffusion(DiffForcingWanModel):
    def __init__(self, clip_path, device, hidden_dim=1024, num_layers=8):
        nn.Module.__init__(self)
        import clip
        wan_model.flash_attention = masked_attention
        encoder,_ = clip.load(str(clip_path), device=device, jit=False)
        encoder.eval().requires_grad_(False)
        # Keep frozen encoder outside DDP parameters/checkpoints.
        self.text_encoder = SimpleNamespace(model=encoder)
        self.text_cache = OrderedDict()
        self.input_dim=32;self.hidden_dim=hidden_dim;self.ffn_dim=hidden_dim*2
        self.freq_dim=256;self.num_heads=8;self.num_layers=num_layers
        self.time_embedding_scale=1.;self.chunk_size=5;self.noise_steps=10
        self.use_text_cond=True;self.drop_out=.1;self.cfg_scale=5.;self.prediction_type='vel'
        self.causal=False;self.text_dim=512;self.text_len=77;self.param_dtype=torch.float32
        self.model=wan_model.WanModel(model_type='t2v',patch_size=(1,1,1),text_len=77,
            in_dim=32,dim=hidden_dim,ffn_dim=self.ffn_dim,freq_dim=256,text_dim=512,out_dim=32,
            num_heads=8,num_layers=num_layers,window_size=(-1,-1),qk_norm=True,cross_attn_norm=True,eps=1e-6,causal=False)

    @torch.no_grad()
    def encode_text_with_cache(self,text_list,device):
        import clip
        missing=list(dict.fromkeys(t for t in text_list if t not in self.text_cache))
        enc=self.text_encoder.model
        if missing:
            tokens=clip.tokenize(missing,truncate=True).to(device)
            x=enc.token_embedding(tokens).to(enc.dtype)+enc.positional_embedding.to(enc.dtype)
            x=enc.transformer(x.permute(1,0,2)).permute(1,0,2)
            x=enc.ln_final(x).float()
            for text,value,length in zip(missing,x,tokens.argmax(-1)+1):
                self.text_cache[text]=value[:int(length)].cpu()
        result=[self.text_cache[t].to(device) for t in text_list]
        for t in text_list:self.text_cache.move_to_end(t)
        while len(self.text_cache)>4096:self.text_cache.popitem(last=False)
        return result


@torch.no_grad()
def encode_motion(vae,features):
    normalized=(features-vae.mean)/vae.std
    padding=(-(normalized.shape[1]-1))%4
    if padding:normalized=torch.cat((normalized,normalized[:,-1:].expand(-1,padding,-1)),1)
    mu,_=vae.model.encode(normalized.transpose(1,2),[0,1],return_dist=True)
    return mu.transpose(1,2)


@torch.no_grad()
def decode_motion(vae,latent):
    return vae.model.decode(latent.transpose(1,2).float(),[0,1]).transpose(1,2)*vae.std+vae.mean
