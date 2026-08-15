#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, statistics, sys, time, traceback
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT/"src"))
import torch, torch.distributed as dist
from spikingbrain_cpu.model_partition import AtlasStage, ArgoStage
from spikingbrain_cpu.protocol import HIDDEN_SHAPE, initialize, shutdown, tensor_summary, validate_hidden
from spikingbrain_cpu.selective_loader import IndexPlanner, RealTensorLoader

def memory():
    p={}; h={}
    for line in Path('/proc/self/status').read_text().splitlines():
        k,_,v=line.partition(':')
        if k in {'VmRSS','VmHWM','VmSwap'}: p[k]=round(int(v.split()[0])/1024,2)
    for line in Path('/proc/meminfo').read_text().splitlines():
        k=line.split(':',1)[0]
        if k in {'MemFree','MemAvailable'}: h[k]=round(int(line.split()[1])/1024,2)
    return {'rss_mib':p.get('VmRSS',0),'hwm_mib':p.get('VmHWM',0),'swap_mib':p.get('VmSwap',0),'mem_free_mib':h.get('MemFree',0),'mem_available_mib':h.get('MemAvailable',0)}

def check_memory(m, limit):
    if m['swap_mib'] != 0: raise MemoryError(f"process swap is nonzero: {m}")
    if m['rss_mib'] > limit or m['hwm_mib'] > limit: raise MemoryError(f"RSS/HWM limit exceeded: {m}")

def synthetic(rank, iterations):
    result={}
    if rank==0:
        small=torch.tensor([1.25,-2.5,7.0]); dist.send(small,1); ack=torch.empty(1); dist.recv(ack,1)
        if ack.item()!=small.sum().item(): raise RuntimeError('small tensor acknowledgement mismatch')
        hidden=torch.arange(3584,dtype=torch.float32).reshape(HIDDEN_SHAPE)/17; echo=torch.empty_like(hidden)
        dist.send(hidden,1); dist.recv(echo,1)
        if not torch.equal(hidden.view(torch.int32),echo.view(torch.int32)): raise RuntimeError('hidden round trip is not byte-identical')
        samples=[]
        for i in range(iterations+5):
            start=time.perf_counter_ns(); dist.send(hidden,1); dist.recv(ack,1); elapsed=(time.perf_counter_ns()-start)/1e6
            if i>=5: samples.append(elapsed)
        result={'small_test':True,'hidden_byte_identical':True,'hidden':tensor_summary(hidden),'iterations':iterations,'round_trip_ms':{'median':statistics.median(samples),'min':min(samples),'max':max(samples)},'estimated_one_way_ms':statistics.median(samples)/2}
    else:
        small=torch.empty(3); dist.recv(small,0); dist.send(small.sum().reshape(1),0)
        hidden=torch.empty(HIDDEN_SHAPE); dist.recv(hidden,0); dist.send(hidden,0)
        ack=torch.ones(1)
        for _ in range(iterations+5): dist.recv(hidden,0); dist.send(ack,0)
        result={'small_test':True,'received_hidden':tensor_summary(hidden)}
    return result

def planner_loader(model_dir,max_bytes,max_rss):
    planner=IndexPlanner.from_files(model_dir/'config.json',model_dir/'model.safetensors.index.json')
    return planner,RealTensorLoader(planner,model_dir,max_bytes,max_rss)

def load_stage(rank,args):
    before=memory(); started=time.perf_counter()
    planner,loader=planner_loader(args.model_dir,args.max_bytes,args.max_rss_mib)
    if rank==0:
        stage=AtlasStage().eval(); stage.load_embedding(loader)
        for i in range(14): stage.load_layer(i,loader)
    else:
        stage=ArgoStage().eval()
        for i in range(14,28): stage.load_layer(i,loader)
        stage.load_final_norm(loader); stage.load_lm_head(loader)
    stage.allocate_quant_buffer(); after=memory(); check_memory(after,args.max_rss_mib)
    return stage,{'before':before,'after':after,'load_ms':(time.perf_counter()-started)*1000,'tensor_count':stage.loaded_tensor_count,'logical_bytes':stage.loaded_logical_bytes,'unique_shards':list(dict.fromkeys(loader.opened_shards))}

def real(rank,args):
    stage,loading=load_stage(rank,args); runs=[]; tokens=[]; previous_rss=None
    token=torch.tensor([[42]],dtype=torch.long)
    for run in range(3):
        gc.collect(); caches=stage.make_decode_caches(position=4096); dist.barrier(); total_start=time.perf_counter()
        if rank==0:
            start=time.perf_counter()
            with torch.inference_mode(): hidden=stage.embed(token); hidden,new_caches,_,_=stage.forward_layers(hidden,caches)
            atlas_ms=(time.perf_counter()-start)*1000; validate_hidden(hidden); boundary=tensor_summary(hidden)
            start=time.perf_counter(); dist.send(hidden,1); send_ms=(time.perf_counter()-start)*1000
            boundary_echo_exact = None
            if run == 0:
                echoed = torch.empty_like(hidden); dist.recv(echoed,1)
                boundary_echo_exact = bool(torch.equal(hidden.view(torch.int32), echoed.view(torch.int32)))
                if not boundary_echo_exact: raise RuntimeError('real boundary echo is not byte-identical')
            returned=torch.empty(1,dtype=torch.int64); start=time.perf_counter(); dist.recv(returned,1); return_wait_ms=(time.perf_counter()-start)*1000
            total_ms=(time.perf_counter()-total_start)*1000; tokens.append(int(returned.item()))
            rec={'run':run+1,'atlas_ms':atlas_ms,'send_ms':send_ms,'boundary_echo_byte_exact':boundary_echo_exact,'return_wait_ms':return_wait_ms,'total_ms':total_ms,'token_id':int(returned.item()),'boundary_sent':boundary,'cache_positions':sorted({c.next_position for c in new_caches.values()}),'memory':memory()}
        else:
            hidden=torch.empty(HIDDEN_SHAPE,dtype=torch.float32); start=time.perf_counter(); dist.recv(hidden,0); recv_ms=(time.perf_counter()-start)*1000
            validate_hidden(hidden); boundary=tensor_summary(hidden)
            if run == 0: dist.send(hidden,0)
            start=time.perf_counter()
            with torch.inference_mode(): hidden,new_caches,_,_=stage.forward_layers(hidden,caches)
            layers_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter(); normalized=stage.apply_final_norm(hidden); norm_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter(); logits=stage.project_logits(normalized); head_ms=(time.perf_counter()-start)*1000
            if not torch.isfinite(logits).all(): raise FloatingPointError('logits contain NaN/Inf')
            start=time.perf_counter(); result=torch.argmax(logits,dim=-1).reshape(1).to(torch.int64); argmax_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter(); dist.send(result,0); return_send_ms=(time.perf_counter()-start)*1000
            tokens.append(int(result.item())); rec={'run':run+1,'recv_ms':recv_ms,'layers_ms':layers_ms,'norm_ms':norm_ms,'lm_head_ms':head_ms,'argmax_ms':argmax_ms,'return_send_ms':return_send_ms,'token_id':int(result.item()),'boundary_received':boundary,'cache_positions':sorted({c.next_position for c in new_caches.values()}),'memory':memory()}
        check_memory(rec['memory'],args.max_rss_mib)
        if previous_rss is not None and rec['memory']['rss_mib'] > previous_rss+512: raise MemoryError('progressive RSS growth exceeds 512 MiB')
        previous_rss=rec['memory']['rss_mib']; runs.append(rec); del caches
    if len(set(tokens))!=1: raise RuntimeError(f'non-repeatable token ids: {tokens}')
    return {'loading':loading,'runs':runs,'repeatable_token':tokens[0],'final_memory':memory()}

def main(default_rank=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,default=default_rank); ap.add_argument('--mode',choices=['synthetic','real'],required=True); ap.add_argument('--master-addr',required=True); ap.add_argument('--port',type=int,default=29500); ap.add_argument('--peer',required=True); ap.add_argument('--iterations',type=int,default=100); ap.add_argument('--model-dir',type=Path); ap.add_argument('--max-bytes',type=int,default=16_000_000_000); ap.add_argument('--max-rss-mib',type=float,default=23552); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--timeout',type=int,default=120); args=ap.parse_args()
    if args.rank not in (0,1):
        ap.error('--rank is required')
    torch.set_num_threads(args.threads)
    initialized=False; report={'rank':args.rank,'mode':args.mode,'success':False}
    try:
        route=initialize(args.rank,args.master_addr,args.port,args.peer,args.timeout); initialized=True; report['network']=route
        report['result']=synthetic(args.rank,args.iterations) if args.mode=='synthetic' else real(args.rank,args)
        report['success']=True
    except Exception as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}; print(json.dumps(report,indent=2),flush=True); raise
    finally:
        if initialized:
            try: shutdown(); report['clean_shutdown']=True
            except Exception as exc: report['clean_shutdown']=False; report['shutdown_error']=str(exc)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
