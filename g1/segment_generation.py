"""Memory-efficient per-position text conditioning without changing historical sources."""
import torch
import numpy as np
from torch import nn
from torch.utils.checkpoint import checkpoint
from models.tools.wan_model import WanModel,WanCrossAttention,sinusoidal_embedding_1d
from g1.generation import G1Diffusion,masked_attention

def pack_context(context,batch,seq_len,device):
    if len(context)==batch:
        contexts=[u for u in context for _ in range(seq_len)]
    elif len(context)==batch*seq_len:contexts=context
    else:raise ValueError(f'Text context count {len(context)} != {batch} or {batch*seq_len}')
    # The upstream frame-wise conditioner repeats tensor references within a span.
    unique=[];groups=[];lookup={}
    for i,value in enumerate(contexts):
        key=(i//seq_len,id(value))
        if key not in lookup:lookup[key]=len(unique);unique.append(value);groups.append([])
        groups[lookup[key]].append(i)
    maximum=max(map(len,groups));indices=torch.zeros(len(groups),maximum,dtype=torch.long)
    valid=torch.zeros_like(indices,dtype=torch.bool)
    for i,group in enumerate(groups):indices[i,:len(group)]=torch.tensor(group);valid[i,:len(group)]=True
    return unique,indices.to(device),valid.to(device)

class SegmentCrossAttention(WanCrossAttention):
    def forward(self,x,context,context_lens):
        if not isinstance(context,tuple):return super().forward(x,context,context_lens)
        embeddings,indices,valid=context
        shape=x.shape;heads=self.num_heads;dim=self.head_dim
        q=self.norm_q(self.q(x)).reshape(-1,heads,dim)[indices]
        k=self.norm_k(self.k(embeddings)).reshape(len(embeddings),-1,heads,dim)
        v=self.v(embeddings).reshape(len(embeddings),-1,heads,dim)
        values=masked_attention(q,k,v,k_lens=context_lens).flatten(2)
        flat=values.new_zeros(shape[0]*shape[1],shape[2])
        flat=flat.index_copy(0,indices[valid],values[valid])
        return self.o(flat.reshape(shape))

class SegmentWanModel(WanModel):
    def forward(self,x,t,context,seq_len,y=None):
        if y is not None:raise ValueError('G1 uses text-to-motion only')
        device=self.patch_embedding.weight.device
        if self.freqs.device!=device:self.freqs=self.freqs.to(device)
        batch=len(x)
        x=[self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes=torch.stack([torch.tensor(u.shape[2:],dtype=torch.long) for u in x])
        x=[u.flatten(2).transpose(1,2) for u in x]
        seq_lens=torch.tensor([u.size(1) for u in x],dtype=torch.long)
        assert seq_lens.max()<=seq_len
        x=torch.cat([torch.cat([u,u.new_zeros(1,seq_len-u.size(1),u.size(2))],1) for u in x])
        if t.dim()==1:t=t.expand(t.size(0),seq_len)
        with torch.autocast('cuda',enabled=False):
            e=self.time_embedding(sinusoidal_embedding_1d(self.freq_dim,t.flatten()).unflatten(0,(batch,seq_len)).float())
            e0=self.time_projection(e).unflatten(2,(6,self.dim))
        unique,indices,valid=pack_context(context,batch,seq_len,device)
        context_lens=torch.tensor([len(u) for u in unique],dtype=torch.long,device=device)
        embedded=self.text_embedding(torch.stack([torch.cat([u,u.new_zeros(self.text_len-len(u),u.size(1))]) for u in unique]))
        for block in self.blocks:
            # Pass the block as a default argument so recomputation binds the correct layer.
            def apply(x,e0,embedded,block=block):
                return block(x,e=e0,seq_lens=seq_lens,grid_sizes=grid_sizes,freqs=self.freqs,
                    context=(embedded,indices,valid),context_lens=context_lens)
            if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                x=checkpoint(apply,x,e0,embedded,use_reentrant=False)
            else:x=apply(x,e0,embedded)
        x=self.head(x,e)
        return [u.float() for u in self.unpatchify(x,grid_sizes)]

class SegmentDiffusion(G1Diffusion):
    def __init__(self,clip_path,device,hidden_dim=1024,num_layers=8,activation_checkpointing=True):
        super().__init__(clip_path,device,hidden_dim,num_layers)
        self.model.__class__=SegmentWanModel
        self.model.activation_checkpointing=activation_checkpointing
        for block in self.model.blocks:block.cross_attn.__class__=SegmentCrossAttention
        self.last_condition_stats={}

    @torch.no_grad()
    def encode_text_with_cache(self,text_list,device):
        # Bound frozen CLIP activations even when a batch has many distinct segments.
        unique=list(dict.fromkeys(text_list));values={}
        for start in range(0,len(unique),64):
            subset=unique[start:start+64]
            values.update(zip(subset,super().encode_text_with_cache(subset,device)))
        return [values[text] for text in text_list]

    def forward(self,batch):
        feature=batch['feature'];lengths=batch['feature_length'];b,seq_len,_=feature.shape
        times=torch.tensor([torch.FloatTensor(1).uniform_(0,int(n)/self.chunk_size).item() for n in lengths],device=feature.device)
        levels=self._get_noise_levels(feature.device,seq_len,times)
        noisy,noise=self.add_noise(feature,levels)
        clean=self.preprocess(feature);noise=self.preprocess(noise);noisy=self.preprocess(noisy)
        ends=[min(int(self.chunk_size*t)+1,int(n)) for t,n in zip(times,lengths)]
        all_text=[];dropped=0;gaps=0;tokens=0;segments=0
        for text,stops,n in zip(batch['text'],batch['feature_text_end'],lengths.tolist()):
            if not text or len(text)!=len(stops) or stops[-1]!=n or any(v<=u for u,v in zip([0]+stops[:-1],stops)):
                raise ValueError('Invalid segment condition boundaries')
            spans=[v-u for u,v in zip([0]+stops[:-1],stops)]
            gaps+=sum(duration for label,duration in zip(text,spans) if not label);tokens+=n;segments+=len(text)
            if np.random.rand()<=self.drop_out:
                all_text.extend(['']*seq_len);dropped+=1
            else:
                for label,duration in zip(text,spans):all_text.extend([label]*duration)
                all_text.extend([text[-1]]*(seq_len-n))
        unique=list(dict.fromkeys(all_text));encoded=self.encode_text_with_cache(unique,feature.device)
        mapping=dict(zip(unique,encoded));contexts=[mapping[t].to(self.param_dtype) for t in all_text]
        pred=self.model([x[:,:n] for x,n in zip(noisy,ends)],levels*self.time_embedding_scale,contexts,seq_len)
        losses=[(p[:,-self.chunk_size:]-(x[:,:n]-eps[:,:n])[:,-self.chunk_size:]).square().sum()
                for p,x,eps,n in zip(pred,clean,noise,ends)]
        loss=torch.stack(losses).mean()
        self.last_condition_stats=dict(examples=b,dropped=dropped,gap_tokens=gaps,tokens=tokens,segments=segments)
        return dict(total=loss,mse=loss)
