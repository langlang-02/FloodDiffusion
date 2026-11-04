"""Check padding attention and frozen VAE latent roundtrip against original API."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from g1.generation import masked_attention,encode_motion,decode_motion
from g1.model import G1CausalVAE

torch.manual_seed(7)
q=torch.randn(2,9,4,16,device='cuda');k=torch.randn(2,11,4,16,device='cuda');v=torch.randn_like(k)
lens=torch.tensor([5,11])
actual=masked_attention(q,k,v,k_lens=lens)
for i,n in enumerate(lens):
    expected=torch.nn.functional.scaled_dot_product_attention(q[i:i+1].transpose(1,2),k[i:i+1,:n].transpose(1,2),v[i:i+1,:n].transpose(1,2)).transpose(1,2)
    torch.testing.assert_close(actual[i:i+1],expected)
c=torch.load('outputs/g1_rot69_root_temporal_500ep_bs2048_mpjpe/best.pt',map_location='cpu',weights_only=False)
m=G1CausalVAE(np.zeros(69,np.float32),np.ones(69,np.float32),**c['identity']['model_params']).cuda().eval().requires_grad_(False)
m.load_state_dict(c['model'])
name=Path('outputs/bones_g1_native38/train.txt').read_text().splitlines()[0]
x=torch.tensor(np.load(f'outputs/bones_g1_rot6d69/motions/{name}.npy'),device='cuda')[None]
with torch.no_grad():
    pred,mu,_=m.reconstruct(x)
    z=encode_motion(m,x)
    torch.testing.assert_close(z,mu.transpose(1,2),rtol=0,atol=0)
    torch.testing.assert_close(decode_motion(m,z)[:,:len(x[0])],pred,rtol=0,atol=0)
assert not z.requires_grad
print('PASS: padded SDPA matches separate sequences; frozen latent encoding/decoding matches original checkpoint API exactly')
