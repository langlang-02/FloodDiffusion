"""Temporal alignment, grouped attention output/gradient parity and FD forward parity."""
import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from g1.segment_data import align_events,validate_events
from g1.segment_generation import SegmentWanModel,SegmentCrossAttention,SegmentDiffusion,pack_context
from g1.generation import masked_attention
from models.tools import wan_model
from models.diffusion_forcing_wan import DiffForcingWanModel

def alignment():
    row=dict(frames=17,start_frame=40)
    events=validate_events([dict(start_time=2.2,end_time=2.6,description='walk'),dict(start_time=2.6,end_time=3.0,description='wave')])
    r=align_events(row,events)
    assert r['text']==['','walk','wave'] and r['token_text_end']==[1,3,5],r
    r=align_events(dict(frames=6,start_frame=0),validate_events([dict(start_time=.25,end_time=.3,description='last')]))
    assert r['text']==['','last'] and r['token_text_end']==[2,3],r
    r=align_events(dict(frames=9,start_frame=0),validate_events([dict(start_time=.01,end_time=.1,description='short')]))
    assert r['covered']==0 and r['short_events']==1
    for bad in [[dict(start_time=0,end_time=0,description='x')],[dict(start_time=0,end_time=2,description='x'),dict(start_time=1,end_time=3,description='y')]]:
        try:validate_events(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid events accepted')
    print('PASS temporal offsets, boundary semantics, gaps, short events and padded tail')

def attention():
    torch.manual_seed(3);wan_model.flash_attention=masked_attention
    ref=wan_model.WanModel(model_type='t2v',patch_size=(1,1,1),text_len=7,in_dim=4,dim=64,ffn_dim=128,freq_dim=16,text_dim=16,out_dim=4,num_heads=4,num_layers=2).cuda()
    with torch.no_grad():ref.head.head.weight.normal_(std=.02)
    grouped=copy.deepcopy(ref);grouped.__class__=SegmentWanModel;grouped.activation_checkpointing=True
    for b in grouped.blocks:b.cross_attn.__class__=SegmentCrossAttention
    xs=[torch.randn(4,n,1,1,device='cuda') for n in [3,5]];t=torch.rand(2,5,device='cuda')
    a,b,c=[torch.randn(n,16,device='cuda') for n in [2,4,6]]
    for contexts in [[a,b],[a,a,b,b,b,c,c,c,a,a]]:
        ref.zero_grad();grouped.zero_grad()
        r=ref(xs,t,contexts,5);g=grouped(xs,t,contexts,5)
        for x,y in zip(r,g):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        sum(x.square().sum() for x in r).backward();sum(x.square().sum() for x in g).backward()
        for (name,p),(other,q) in zip(ref.named_parameters(),grouped.named_parameters()):
            assert name==other
            torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4,msg=lambda m:f'{name}: {m}')
    _,indices,_=pack_context([a]*100,20,5,torch.device('cuda'))
    assert indices.shape==(20,5) # Never create a giant shared blank-text query group.
    ca=copy.deepcopy(ref.blocks[0].cross_attn);ca.__class__=SegmentCrossAttention
    x=torch.randn(2,5,64,device='cuda');context=torch.randn(4,7,64,device='cuda')
    index=torch.tensor([[0,1,0],[2,3,4],[5,6,0],[7,8,9]],device='cuda');valid=torch.tensor([[1,1,0],[1,1,1],[1,1,0],[1,1,1]],device='cuda',dtype=torch.bool)
    lengths=torch.full((4,),7,device='cuda');first=ca(x,(context,index,valid),lengths)
    changed=context.clone();changed[1]+=2;second=ca(x,(changed,index,valid),lengths)
    torch.testing.assert_close(first[1],second[1]);torch.testing.assert_close(first[0,:2],second[0,:2]);assert not torch.allclose(first[0,2:],second[0,2:])
    print('PASS full Wan single/multiple condition output+gradient parity, checkpointing, sample and interval isolation')

def forward():
    model=SegmentDiffusion('/data/langlang/cache/clip/ViT-B-32.pt',torch.device('cuda'),128,2).cuda()
    batch=dict(feature=torch.randn(2,12,32,device='cuda'),feature_length=torch.tensor([10,12],device='cuda'),text=[['walk','wave'],['','sit']],feature_text_end=[[5,10],[3,12]])
    with torch.no_grad():model.model.head.head.weight.normal_(std=.01)
    for seed in range(4):
        torch.manual_seed(seed);np.random.seed(seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):reference=DiffForcingWanModel.forward(model,batch)['total']
        torch.manual_seed(seed);np.random.seed(seed)
        with torch.autocast('cuda',dtype=torch.bfloat16):actual=model(batch)['total']
        torch.testing.assert_close(reference,actual,rtol=1e-5,atol=1e-4)
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        result=model.generate(dict(feature_length=torch.tensor([10,12],device='cuda'),text=batch['text'],feature_text_end=batch['feature_text_end']))
    assert [v.shape for v in result['generated']]==[torch.Size([10,32]),torch.Size([12,32])]
    assert all(torch.isfinite(v).all() for v in result['generated'])
    print('PASS original diffusion-forcing objective parity and variable-length multisegment generation')

if __name__=='__main__':alignment();attention();forward()
