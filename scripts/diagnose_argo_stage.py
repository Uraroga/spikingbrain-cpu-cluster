#!/usr/bin/env python3
"""One real, local ArgoStage forward for independent-process diagnosis."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent)); sys.path.insert(0,'/app/scripts')
import torch
from distributed_stage import load_stage, memory

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-dir',type=Path,required=True); p.add_argument('--max-bytes',type=int,default=15384997376); p.add_argument('--max-rss-mib',type=float,default=23552); p.add_argument('--threads',type=int,default=4); a=p.parse_args(); a.rank=1
    torch.set_num_threads(a.threads); start=time.perf_counter(); stage,loading=load_stage(1,a); load_ms=(time.perf_counter()-start)*1000
    hidden=torch.linspace(-1,1,3584).reshape(1,1,3584); start=time.perf_counter()
    with torch.inference_mode():
        hidden,caches,_,_=stage.forward_layers(hidden,{})
        hidden=stage.apply_final_norm(hidden); logits=stage.project_logits(hidden); token=torch.argmax(logits[:,-1,:])
    elapsed=(time.perf_counter()-start)*1000
    print(json.dumps({'torch':torch.__version__,'capability':torch.backends.cpu.get_cpu_capability(),'env':{k:v for k,v in os.environ.items() if any(x in k.upper() for x in ('ATEN','MKL','DNNL','OMP'))},'loading':loading,'load_ms':load_ms,'forward_ms':elapsed,'finite':bool(torch.isfinite(logits).all()),'token':int(token),'cache_positions':sorted({c.next_position for c in caches.values()}),'memory':memory()},indent=2))
if __name__=='__main__': main()
