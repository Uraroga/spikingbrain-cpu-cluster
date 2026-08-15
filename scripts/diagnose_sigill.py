#!/usr/bin/env python3
"""Bounded, single-case CPU microtests for SIGILL diagnosis."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.nn.functional as F
from spikingbrain_cpu.block import GLABlock, QuantBuffer, SlidingWindowAttentionBlock
from spikingbrain_cpu.ops import RMSNorm, gla_recurrent, sliding_window_attention

def run(case,reps):
    torch.manual_seed(85); result=None
    if case=='exp':
        x=torch.linspace(-10,10,1_000_003)
        for _ in range(reps): result=torch.exp(x)
    elif case=='exp_logsigmoid':
        x=torch.linspace(-10,10,1_000_003)
        for _ in range(reps):
            result=torch.exp(x)
            result=F.logsigmoid(result.mul_(0.01))
    elif case=='logsigmoid':
        x=torch.linspace(-10,10,1_000_003)
        for _ in range(reps): result=F.logsigmoid(x)
    elif case.startswith('linear_') or case=='lm_head':
        dims={'linear_small':(64,64),'linear_3584_3584':(3584,3584),'linear_3584_18944':(3584,18944),'linear_18944_3584':(18944,3584),'lm_head':(3584,152064)}
        i,o=dims[case]; x=torch.randn(1,i); w=torch.randn(o,i)
        for _ in range(reps): result=F.linear(x,w)
    elif case=='fake_quant':
        x=torch.randn(4096,4096); scale=torch.rand(4096,1)+0.01
        for _ in range(reps): result=torch.round(x/scale).clamp(-128,127)*scale
    elif case=='rmsnorm':
        op=RMSNorm(3584); x=torch.randn(8,3584)
        for _ in range(reps): result=op(x)
    elif case=='attention':
        q=torch.randn(1,28,8,128); k=torch.randn(1,28,8,128); v=torch.randn(1,28,8,128)
        for _ in range(reps): result=sliding_window_attention(q,k,v,4096)
    elif case=='gla':
        q=torch.randn(1,28,8,128); k=torch.randn_like(q); v=torch.randn_like(q); g=F.logsigmoid(torch.randn_like(q))/16
        for _ in range(reps): result,_=gla_recurrent(q,k,v,g)
    elif case in {'gla_layer','attention_layer'}:
        buffer=QuantBuffer(); block=(GLABlock(buffer=buffer) if case=='gla_layer' else SlidingWindowAttentionBlock(buffer=buffer)).eval(); x=torch.randn(1,1,3584)
        with torch.inference_mode():
            for _ in range(reps): result,_=block(x)
    else: raise ValueError(case)
    return {'shape':list(result.shape),'dtype':str(result.dtype),'finite':bool(torch.isfinite(result).all()),'checksum':float(result.reshape(-1)[::1024].sum())}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--case',required=True); p.add_argument('--reps',type=int,default=3); p.add_argument('--threads',type=int,default=4); a=p.parse_args(); torch.set_num_threads(a.threads)
    started=time.perf_counter(); result=run(a.case,a.reps)
    print(json.dumps({'case':a.case,'reps':a.reps,'threads':torch.get_num_threads(),'capability':torch.backends.cpu.get_cpu_capability(),'env':{k:v for k,v in os.environ.items() if any(x in k.upper() for x in ('ATEN','MKL','DNNL','OMP'))},'elapsed_ms':(time.perf_counter()-started)*1000,'result':result},indent=2))
if __name__=='__main__': main()
