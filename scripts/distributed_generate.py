#!/usr/bin/env python3
"""One short distributed greedy generation session."""
from __future__ import annotations
import argparse, json, os, statistics, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
import torch, torch.distributed as dist
from distributed_stage import load_stage, memory, check_memory
from spikingbrain_cpu.protocol import initialize, shutdown, tensor_summary, validate_hidden

def cache_report(caches):
    return {'count':len(caches),'positions':sorted({c.next_position for c in caches.values()}),'bytes':sum(c.nbytes for c in caches.values()),'types':{str(i):type(c).__name__ for i,c in caches.items()}}

def ensure_caches(caches, indices, position):
    if set(caches)!=set(indices): raise RuntimeError(f'cache indices mismatch: {sorted(caches)}')
    positions={c.next_position for c in caches.values()}
    if positions!={position}: raise RuntimeError(f'cache position mismatch: {positions}, expected {position}')

def send_hidden(hidden, peer):
    validate_hidden(hidden) if hidden.shape[1]==1 else None
    header=torch.tensor([hidden.shape[1]],dtype=torch.int64); dist.send(header,peer)
    start=time.perf_counter(); dist.send(hidden.contiguous(),peer); return (time.perf_counter()-start)*1000

def recv_hidden(peer,max_length):
    header=torch.empty(1,dtype=torch.int64); dist.recv(header,peer); length=int(header.item())
    if not 1<=length<=max_length: raise ValueError(f'invalid hidden length {length}; limit {max_length}')
    hidden=torch.empty(1,length,3584,dtype=torch.float32); start=time.perf_counter(); dist.recv(hidden,peer); elapsed=(time.perf_counter()-start)*1000
    if hidden.dtype!=torch.float32 or not torch.isfinite(hidden).all(): raise FloatingPointError('invalid received hidden')
    return hidden,elapsed

def atlas(args,stage):
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    from transformers import Qwen2Tokenizer
    start=time.perf_counter(); tokenizer=Qwen2Tokenizer.from_pretrained(str(args.tokenizer_dir),local_files_only=True); tokenizer_load_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    if args.chat_user_message is not None:
        messages=[{'role':'user','content':args.chat_user_message}]
        rendered_prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        prompt_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        prompt_mode='chat_template'
    else:
        messages=None; rendered_prompt=args.prompt
        prompt_ids=tokenizer.encode(args.prompt,add_special_tokens=False)
        prompt_mode='raw_completion'
    tokenize_ms=(time.perf_counter()-start)*1000
    if not 0<len(prompt_ids)<=args.max_prompt_tokens: raise ValueError(f'prompt has {len(prompt_ids)} tokens; limit {args.max_prompt_tokens}')
    if any(not 0<=x<152064 for x in prompt_ids): raise ValueError('prompt token outside model vocabulary')
    roundtrip=tokenizer.decode(prompt_ids,skip_special_tokens=False)
    ids=torch.tensor([prompt_ids],dtype=torch.long); total_start=time.perf_counter(); caches={}
    start=time.perf_counter()
    with torch.inference_mode(): hidden=stage.embed(ids); hidden,caches,_,_=stage.forward_layers(hidden,caches)
    if not torch.isfinite(hidden).all(): raise FloatingPointError('atlas hidden contains NaN/Inf after prefill')
    prefill_ms=(time.perf_counter()-start)*1000; ensure_caches(caches,range(14),len(prompt_ids)); boundary=tensor_summary(hidden[:,-1:])
    prefill_send_ms=send_hidden(hidden,1); generated=[]; steps=[]; cache_states=[{'event':'after_prefill',**cache_report(caches)}]
    while len(generated)<args.max_new_tokens:
        received=torch.empty(1,dtype=torch.int64); start=time.perf_counter(); dist.recv(received,1); token_wait_ms=(time.perf_counter()-start)*1000; token=int(received.item())
        if not 0<=token<len(tokenizer): raise ValueError(f'generated token outside tokenizer range: {token} >= {len(tokenizer)}')
        generated.append(token); stop=(token==tokenizer.eos_token_id or len(generated)>=args.max_new_tokens)
        dist.send(torch.tensor([0 if stop else 1],dtype=torch.int64),1)
        step={'generated_index':len(generated),'token_id':token,'token_wait_ms':token_wait_ms,'stopped':stop}
        if not stop:
            start=time.perf_counter()
            with torch.inference_mode(): next_hidden=stage.embed(torch.tensor([[token]])); next_hidden,caches,_,_=stage.forward_layers(next_hidden,caches)
            if not torch.isfinite(next_hidden).all(): raise FloatingPointError('atlas hidden contains NaN/Inf during decode')
            step['atlas_decode_ms']=(time.perf_counter()-start)*1000; ensure_caches(caches,range(14),len(prompt_ids)+len(generated)); step['send_ms']=send_hidden(next_hidden,1); cache_states.append({'event':f'after_processing_generated_{len(generated)}',**cache_report(caches)})
        steps.append(step)
        if stop: break
    total_ms=(time.perf_counter()-total_start)*1000; full=prompt_ids+generated
    return {'prompt':args.prompt if messages is None else args.chat_user_message,'prompt_mode':prompt_mode,'messages':messages,'rendered_prompt':rendered_prompt,'prompt_ids':prompt_ids,'prompt_length':len(prompt_ids),'prompt_roundtrip':roundtrip,'generated_ids':generated,'full_text':tokenizer.decode(full,skip_special_tokens=False),'continuation':tokenizer.decode(generated,skip_special_tokens=False),'tokenizer':{'class':type(tokenizer).__name__,'vocab_size_property':tokenizer.vocab_size,'length':len(tokenizer),'bos_token':tokenizer.bos_token,'bos_token_id':tokenizer.bos_token_id,'eos_token':tokenizer.eos_token,'eos_token_id':tokenizer.eos_token_id,'pad_token':tokenizer.pad_token,'pad_token_id':tokenizer.pad_token_id,'add_bos_token':bool(tokenizer.init_kwargs.get('add_bos_token',False)),'chat_template_present':bool(tokenizer.chat_template),'load_ms':tokenizer_load_ms,'tokenize_ms':tokenize_ms},'prefill_atlas_ms':prefill_ms,'prefill_send_ms':prefill_send_ms,'boundary_last_token':boundary,'steps':steps,'cache_states':cache_states,'cache_object_policy':'state returned by each layer is fed directly to the next step; caches are never reset within the session','numerical_checks':{'atlas_hidden_finite_each_step':True,'generated_ids_in_tokenizer_range':True},'total_generation_ms':total_ms,'stop_reason':'eos' if generated[-1]==tokenizer.eos_token_id else 'max_new_tokens','memory':memory()}

def argo(args,stage):
    caches={}; generated=[]; steps=[]; cache_states=[]; prompt_length=None
    while len(generated)<args.max_new_tokens:
        hidden,recv_ms=recv_hidden(0,args.max_prompt_tokens)
        if prompt_length is None: prompt_length=hidden.shape[1]
        expected=prompt_length+len(generated)
        start=time.perf_counter()
        with torch.inference_mode(): hidden,caches,_,_=stage.forward_layers(hidden,caches)
        layers_ms=(time.perf_counter()-start)*1000; ensure_caches(caches,range(14,28),expected)
        if not torch.isfinite(hidden).all(): raise FloatingPointError('argo hidden contains NaN/Inf')
        start=time.perf_counter(); normalized=stage.apply_final_norm(hidden[:,-1:]); norm_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter(); logits=stage.project_logits(normalized); head_ms=(time.perf_counter()-start)*1000
        if not torch.isfinite(logits).all(): raise FloatingPointError('non-finite logits')
        start=time.perf_counter(); token=torch.argmax(logits[:,-1,:],dim=-1).to(torch.int64); argmax_ms=(time.perf_counter()-start)*1000
        dist.send(token.reshape(1),0); generated.append(int(token.item())); cache_states.append({'event':'after_prefill' if len(generated)==1 else f'after_processing_generated_{len(generated)-1}',**cache_report(caches)})
        control=torch.empty(1,dtype=torch.int64); dist.recv(control,0)
        steps.append({'generated_index':len(generated),'token_id':generated[-1],'recv_ms':recv_ms,'layers_ms':layers_ms,'norm_ms':norm_ms,'lm_head_ms':head_ms,'argmax_ms':argmax_ms,'continue':bool(control.item()),'memory':memory()})
        check_memory(steps[-1]['memory'],args.max_rss_mib)
        if not control.item(): break
    return {'generated_ids':generated,'prompt_length':prompt_length,'steps':steps,'cache_states':cache_states,'cache_object_policy':'state returned by each layer is fed directly to the next step; caches are never reset within the session','numerical_checks':{'received_hidden_finite_each_step':True,'argo_hidden_finite_each_step':True,'logits_finite_each_step':True,'generated_ids_in_model_range':True},'memory':memory()}

def main(default_rank=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,default=default_rank); ap.add_argument('--master-addr',required=True); ap.add_argument('--port',type=int,default=29500); ap.add_argument('--peer',required=True); ap.add_argument('--model-dir',type=Path,required=True); ap.add_argument('--tokenizer-dir',type=Path); ap.add_argument('--prompt',default='Hello'); ap.add_argument('--chat-user-message'); ap.add_argument('--max-prompt-tokens',type=int,default=8); ap.add_argument('--max-new-tokens',type=int,default=3); ap.add_argument('--max-bytes',type=int,required=True); ap.add_argument('--max-rss-mib',type=float,default=23552); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--timeout',type=int,default=180); args=ap.parse_args()
    torch.set_num_threads(args.threads); initialized=False; report={'rank':args.rank,'success':False}
    try:
        report['network']=initialize(args.rank,args.master_addr,args.port,args.peer,args.timeout); initialized=True
        stage,loading=load_stage(args.rank,args); report['loading']=loading; report['result']=atlas(args,stage) if args.rank==0 else argo(args,stage); report['success']=True
    except Exception as exc: report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}; print(json.dumps(report,indent=2),flush=True); raise
    finally:
        if initialized:
            try: shutdown(); report['clean_shutdown']=True
            except Exception as exc: report['clean_shutdown']=False; report['shutdown_error']=str(exc)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
