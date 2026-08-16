#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, json, math, os, platform, random, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import sys
sys.path.insert(0, "/app/src")
from spikingbrain_cpu.block import GLABlock
from spikingbrain_cpu.ops import dynamic_spikes, quant_linear
from spikingbrain_cpu.selective_loader import IndexPlanner, RealTensorLoader, materialize_layer

NAME = "model.layers.0.mlp.gate_proj"

def mem():
    p={}
    for line in Path('/proc/self/status').read_text().splitlines():
        k,_,v=line.partition(':')
        if k in {'VmRSS','VmHWM','VmSwap'}: p[k]=round(int(v.split()[0])/1024,2)
    return {'rss_mib':p.get('VmRSS',0),'hwm_mib':p.get('VmHWM',0),'swap_mib':p.get('VmSwap',0)}

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8<<20): h.update(b)
    return h.hexdigest()

def load_pair(model_dir: Path):
    planner=IndexPlanner.from_files(model_dir/'config.json',model_dir/'model.safetensors.index.json')
    names=(NAME+'.weight',NAME+'.weight_quantizer.scales')
    plans={p.name:p for p in planner.all_tensors()}
    loader=RealTensorLoader(planner,model_dir,sum(plans[n].nbytes for n in names),24000)
    vals={p.name:t for p,t in loader.iter_materialized(names)}
    return vals[names[0]],vals[names[1]],plans,loader.opened_shards

def capture(args):
    from transformers import AutoTokenizer
    planner=IndexPlanner.from_files(args.model_dir/'config.json',args.model_dir/'model.safetensors.index.json')
    block=GLABlock(device='meta').eval()
    loader=RealTensorLoader(planner,args.model_dir,1_600_000_000,24000)
    materialize_layer(block,0,loader)
    emb_name='model.embeddings.weight'; emb_plan={p.name:p for p in planner.all_tensors()}[emb_name]
    emb_loader=RealTensorLoader(planner,args.model_dir,emb_plan.nbytes,24000)
    _,embedding=next(emb_loader.iter_materialized((emb_name,)))
    tok=AutoTokenizer.from_pretrained(args.tokenizer_dir,local_files_only=True)
    text=("Spiking neural networks and efficient language models can share sparse computation, "
          "but CPU inference must preserve numerical behavior across legacy processors. ")*4
    ids=tok(text,add_special_tokens=False)['input_ids'][:32]
    if len(ids)!=32: raise RuntimeError(f'expected 32 tokens, got {len(ids)}')
    hidden=F.embedding(torch.tensor(ids).reshape(1,-1),embedding)
    torch.set_num_threads(args.threads)
    with torch.inference_mode():
        residual=hidden
        normalized=block.attn_norm(hidden)
        attention_output,_=block._attention(normalized,None)
        gate_input,_=block.mlp_norm(attention_output,residual,prenorm=True)
    record={'inputs':gate_input.reshape(32,3584).contiguous(),'token_ids':ids,'text':text,
            'source':'layer-0 mlp_norm output after real embedding and layer-0 GLA attention',
            'checkpoint_config_sha256':sha256(args.model_dir/'config.json')}
    torch.save(record,args.inputs)
    print(json.dumps({'capture':str(args.inputs),'shape':list(record['inputs'].shape),'token_ids':ids,
                      'finite':bool(torch.isfinite(record['inputs']).all()),'memory':mem()},indent=2))

def pct(xs,p):
    ys=sorted(xs); pos=(len(ys)-1)*p; lo=int(pos); hi=min(lo+1,len(ys)-1); f=pos-lo
    return ys[lo]*(1-f)+ys[hi]*f

def stats(xs):
    return {'n':len(xs),'median_ms':statistics.median(xs),'p10_ms':pct(xs,.1),'p90_ms':pct(xs,.9),'min_ms':min(xs)}

def bootstrap_ci(xs,seed=1313):
    r=random.Random(seed); meds=[]; n=len(xs)
    for _ in range(2000): meds.append(statistics.median(xs[r.randrange(n)] for __ in range(n)))
    return [pct(meds,.025),pct(meds,.975)]

def err(reference,actual):
    d=(actual-reference).double(); ref=reference.double()
    denom=torch.linalg.vector_norm(ref).item()
    return {'max_abs':d.abs().max().item(),'mean_abs':d.abs().mean().item(),
            'rmse':d.square().mean().sqrt().item(),
            'relative_l2':torch.linalg.vector_norm(d).item()/denom if denom else float('nan'),
            'cosine':F.cosine_similarity(ref.reshape(1,-1),actual.double().reshape(1,-1)).item(),
            'top_error_indices':torch.topk(d.abs(),5).indices.tolist()}

def benchmark(args):
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    ext=load(name='goal13b_int8_kernel',sources=[str(args.kernel)],extra_cflags=['-O3','-msse4.1','-fopenmp'],extra_ldflags=['-fopenmp'],verbose=False)
    pre={'start_memory':mem()}
    t=time.perf_counter(); weight,scales,plans,shards=load_pair(args.model_dir); pre['load_ms']=(time.perf_counter()-t)*1000
    weight=weight.contiguous(); scales=scales.squeeze(-1).contiguous(); O,K=weight.shape; G=K//128
    inputs_record=torch.load(args.inputs,map_location='cpu',weights_only=True); inputs=inputs_record['inputs'].float().contiguous()
    synthetic=torch.randn(3584,generator=torch.Generator().manual_seed(1313)); all_inputs=torch.cat((inputs,synthetic[None]),0)
    pre['after_load_memory']=mem()
    t=time.perf_counter(); codes=torch.round(weight.reshape(O,G,128)/scales.unsqueeze(-1)); prep_codes=(time.perf_counter()-t)*1000
    qmin,qmax=int(codes.min()),int(codes.max()); mask=(codes < -128)|(codes > 127)
    t=time.perf_counter(); base=codes.clamp(-128,127).to(torch.int8).reshape(O,K).contiguous(); prep_base=(time.perf_counter()-t)*1000
    nz=mask.nonzero(); row=nz[:,0]; kval=nz[:,1]*128+nz[:,2]
    residual=(codes[mask]-codes[mask].clamp(-128,127)).to(torch.int16).contiguous()
    counts=torch.bincount(row,minlength=O); rowptr=torch.cat((torch.zeros(1,dtype=torch.int64),counts.cumsum(0))).contiguous()
    indices=kval.to(torch.int32).contiguous()
    prep_total=prep_codes+prep_base
    wfq=(codes*scales.unsqueeze(-1)).reshape(O,K).contiguous()
    buffer=torch.empty_like(weight)
    pre.update({'code_prepare_ms':prep_codes,'base_pack_ms':prep_base,'preparation_ms':prep_total,
                'code_range':[qmin,qmax],'weight_outliers':int(mask.sum()),'after_prepare_memory':mem()})
    decoded=base.to(torch.int16).reshape(O,G,128)
    decoded.view(-1)[(row*K+kval).long()] += residual
    if not torch.equal(decoded,codes.to(torch.int16)): raise RuntimeError('B2 decoded weight codes are not exact')
    use_avx2='avx2' in Path('/proc/cpuinfo').read_text().split('flags',1)[1].split('\n',1)[0].split()

    components={k:[] for k in ['B1_activation','B1_kernel','B1_scale','B2_activation','B2_kernel','B2_outlier','B2_scale']}
    def acode(x):
        sp,th=dynamic_spikes(x); return sp.reshape(-1),th.reshape(())
    def A(x): return quant_linear(x,weight,scales.unsqueeze(-1),None,128,buffer)
    def C(x):
        sp,th=dynamic_spikes(x); return F.linear((sp*th).to(x.dtype),wfq)
    def B1(x,record=False):
        t=time.perf_counter(); a,th=acode(x); a8=a.clamp(-128,127).to(torch.int8).contiguous(); t1=time.perf_counter()
        acc=ext.group_acc_i8(base,a8,use_avx2); t2=time.perf_counter()
        y=(acc.float()*scales).sum(1)*th; t3=time.perf_counter()
        if record: components['B1_activation'].append((t1-t)*1000);components['B1_kernel'].append((t2-t1)*1000);components['B1_scale'].append((t3-t2)*1000)
        return y
    def B2(x,record=False):
        t=time.perf_counter(); a,th=acode(x); a16=a.to(torch.int16).contiguous(); t1=time.perf_counter()
        acc=ext.group_acc_i16(base,a16,use_avx2); t2=time.perf_counter()
        corr=ext.outlier_correction(rowptr,indices,residual,a16,G); t3=time.perf_counter()
        y=((acc+corr).float()*scales).sum(1)*th; t4=time.perf_counter()
        if record: components['B2_activation'].append((t1-t)*1000);components['B2_kernel'].append((t2-t1)*1000);components['B2_outlier'].append((t3-t2)*1000);components['B2_scale'].append((t4-t3)*1000)
        return y
    paths={'A':A,'B1':B1,'B2':B2,'C':C}; x0=inputs[0]
    cold={}
    for n,f in paths.items(): t=time.perf_counter(); y=f(x0); cold[n]=(time.perf_counter()-t)*1000; float(y.sum())
    for _ in range(10):
        order=list(paths); random.Random(1313+_).shuffle(order)
        for n in order: float(paths[n](x0).sum())
    samples={n:[] for n in paths}; elapsed={n:0.0 for n in paths}; rng=random.Random(1313); idx=0
    while any(len(samples[n])<30 or elapsed[n]<2.0 for n in paths):
        order=[n for n in paths if len(samples[n])<30 or elapsed[n]<2.0]; rng.shuffle(order); x=inputs[idx%len(inputs)]; idx+=1
        for n in order:
            t=time.perf_counter(); y=paths[n](x,True) if n in {'B1','B2'} else paths[n](x); dt=time.perf_counter()-t
            float(y.sum()); samples[n].append(dt*1000); elapsed[n]+=dt
            if len(samples[n])>=1000: elapsed[n]=999
    timing={n:{**stats(v),'median_bootstrap_95ci_ms':bootstrap_ci(v),'measured_seconds':sum(v)/1000} for n,v in samples.items()}
    timing['cold_ms']=cold
    timing['components']={n:stats(v) for n,v in components.items()}
    activation=[]; numerical={'B1':[],'B2':[],'C':[]}
    with torch.inference_mode():
        for i,x in enumerate(all_inputs):
            a,th=acode(x); activation.append({'input':i,'min':int(a.min()),'max':int(a.max()),'outside_int8':int(((a < -128)|(a>127)).sum()),'threshold':float(th)})
            ref=A(x); numerical['B1'].append(err(ref,B1(x))); numerical['B2'].append(err(ref,B2(x))); numerical['C'].append(err(ref,C(x)))
    logical={'fp32_weight':weight.nbytes,'fp32_fake_quant_buffer':buffer.nbytes,'fp32_dequantized_reference':wfq.nbytes,
             'fp32_scales':scales.nbytes,'int8_base':base.nbytes,'outlier_rowptr':rowptr.nbytes,
             'outlier_indices':indices.nbytes,'outlier_residuals':residual.nbytes,'int32_group_scratch':O*G*4,
             'int16_activation':K*2,'int8_activation':K}
    pre['before_release_memory']=mem()
    deployment_bytes=logical['int8_base']+logical['fp32_scales']+logical['outlier_rowptr']+logical['outlier_indices']+logical['outlier_residuals']+logical['int32_group_scratch']+logical['int16_activation']
    result={'environment':{'hostname':platform.node(),'cpu':platform.processor(),'torch':torch.__version__,'threads':args.threads,
             'interop_threads':torch.get_num_interop_threads(),'affinity':sorted(os.sched_getaffinity(0)),'avx2_kernel':use_avx2,
             'image_id':args.image_id},'matrix':{'name':NAME,'shape':[O,K],'shards':shards,
             'config_sha256':sha256(args.model_dir/'config.json'),'index_sha256':sha256(args.model_dir/'model.safetensors.index.json')},
             'preparation':pre,'activation_codes':activation,'timing':timing,'logical_bytes':logical,
             'deployment_B2_bytes':deployment_bytes,'numerical':numerical,'checksum':float(B2(x0).sum()),
             'b2_code_exact':True,'synthetic_input_index':len(all_inputs)-1}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'timing':timing,'deployment_B2_bytes':deployment_bytes,'checksum':result['checksum']},indent=2))

def deployment_memory(args):
    torch.set_num_threads(args.threads)
    before=mem(); weight,scales,plans,shards=load_pair(args.model_dir); after_load=mem()
    O,K=weight.shape; G=K//128
    t=time.perf_counter(); codes=torch.round(weight.reshape(O,G,128)/scales.squeeze(-1).unsqueeze(-1)); mask=(codes < -128)|(codes > 127)
    base=codes.clamp(-128,127).to(torch.int8).reshape(O,K).contiguous(); nz=mask.nonzero(); row= nz[:,0]; kval=nz[:,1]*128+nz[:,2]
    residual=(codes[mask]-codes[mask].clamp(-128,127)).to(torch.int16).contiguous(); counts=torch.bincount(row,minlength=O)
    rowptr=torch.cat((torch.zeros(1,dtype=torch.int64),counts.cumsum(0))).contiguous(); indices=kval.to(torch.int32).contiguous(); scales2=scales.squeeze(-1).contiguous()
    prepare_ms=(time.perf_counter()-t)*1000; after_prepare=mem()
    torch.save({'base':base,'scales':scales2,'rowptr':rowptr,'indices':indices,'residuals':residual},args.packed)
    serialized_bytes=args.packed.stat().st_size
    del weight,scales,codes,mask,nz,row,kval,counts; gc.collect(); after_release=mem()
    logical=sum(x.nbytes for x in (base,scales2,rowptr,indices,residual))
    result={'hostname':platform.node(),'threads':args.threads,'before':before,'after_load':after_load,'after_prepare':after_prepare,
            'after_fp32_source_release':after_release,'prepare_ms':prepare_ms,'logical_persistent_bytes':logical,
            'serialized_bytes':serialized_bytes,'packed_sha256':sha256(args.packed),'outliers':residual.numel(),'shards':shards}
    args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    c=sub.add_parser('capture'); c.add_argument('--model-dir',type=Path,required=True);c.add_argument('--tokenizer-dir',type=Path,required=True);c.add_argument('--inputs',type=Path,required=True);c.add_argument('--threads',type=int,default=4)
    b=sub.add_parser('benchmark');b.add_argument('--model-dir',type=Path,required=True);b.add_argument('--inputs',type=Path,required=True);b.add_argument('--kernel',type=Path,required=True);b.add_argument('--output',type=Path,required=True);b.add_argument('--threads',type=int,required=True);b.add_argument('--image-id',required=True)
    d=sub.add_parser('deployment-memory');d.add_argument('--model-dir',type=Path,required=True);d.add_argument('--output',type=Path,required=True);d.add_argument('--packed',type=Path,required=True);d.add_argument('--threads',type=int,required=True)
    a=p.parse_args(); capture(a) if a.mode=='capture' else deployment_memory(a) if a.mode=='deployment-memory' else benchmark(a)
if __name__=='__main__': main()
