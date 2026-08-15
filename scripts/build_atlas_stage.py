#!/usr/bin/env python3
"""Incrementally load and benchmark the real atlas5 partition."""
from __future__ import annotations
import argparse, json, platform, statistics, sys, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import torch
from spikingbrain_cpu.model_partition import AtlasStage
from spikingbrain_cpu.selective_loader import IndexPlanner, LoaderLimitError, RealTensorLoader

def proc():
    data = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmHWM", "VmSwap"}: data[key] = round(int(value.split()[0])/1024, 2)
    return {"rss_mib":data.get("VmRSS",0), "hwm_mib":data.get("VmHWM",0), "swap_mib":data.get("VmSwap",0)}

def host():
    data = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key = line.split(":",1)[0]
        if key in {"MemFree","MemAvailable","SwapTotal","SwapFree"}: data[key]=round(int(line.split()[1])/1024,2)
    return {"free_mib":data.get("MemFree",0),"available_mib":data.get("MemAvailable",0),"swap_total_mib":data.get("SwapTotal",0),"swap_free_mib":data.get("SwapFree",0)}

def guard(limit, floor, where):
    p, h = proc(), host()
    if p["rss_mib"] > limit or p["hwm_mib"] > limit: raise LoaderLimitError(f"RSS limit at {where}: {p}")
    if p["swap_mib"]: raise LoaderLimitError(f"process swap at {where}: {p['swap_mib']} MiB")
    if h["available_mib"] < floor: raise LoaderLimitError(f"host RAM floor at {where}: {h}")

def touch(module):
    with torch.inference_mode(): return sum(x.view(-1)[::1024].sum().item() for x in list(module.parameters())+list(module.buffers()))

def measure(fn, warmup=1, iterations=3):
    with torch.inference_mode():
        for _ in range(warmup): fn()
        values=[]; last=None
        for _ in range(iterations):
            start=time.perf_counter(); last=fn(); values.append((time.perf_counter()-start)*1000)
    return {"median_ms":round(statistics.median(values),3),"min_ms":round(min(values),3),"max_ms":round(max(values),3),"warmup":warmup,"iterations":iterations}, last

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",type=Path,required=True); ap.add_argument("--max-bytes",type=int,required=True); ap.add_argument("--max-rss-mib",type=float,required=True); ap.add_argument("--min-host-available-mib",type=float,default=4096); ap.add_argument("--threads",type=int,default=4); a=ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(606)
    planner=IndexPlanner.from_files(a.model_dir/"config.json",a.model_dir/"model.safetensors.index.json")
    atlas,_=planner.plan_split(14)
    if len(atlas.tensors)!=295 or atlas.nbytes!=15_384_983_040: raise RuntimeError(f"unexpected atlas plan {len(atlas.tensors)} {atlas.nbytes}")
    if atlas.nbytes>a.max_bytes: raise LoaderLimitError("planned bytes exceed limit")
    initial={"process":proc(),"host":host()}; stage=AtlasStage().eval(); meta=proc(); loader=RealTensorLoader(planner,a.model_dir,a.max_bytes,a.max_rss_mib)
    started=time.perf_counter(); stage.load_embedding(loader); embedding_checksum=touch(stage.embeddings); checkpoints=[{"layers":0,"tensor_count":stage.loaded_tensor_count,"logical_bytes":stage.loaded_logical_bytes,"rss":proc(),"host":host(),"load_ms":round((time.perf_counter()-started)*1000,3)}]; guard(a.max_rss_mib,a.min_host_available_mib,"embedding")
    cumulative=time.perf_counter(); sampled=0.0
    for count,index in enumerate(range(14),1):
        stage.load_layer(index,loader); sampled+=touch(stage.layers[str(index)]); guard(a.max_rss_mib,a.min_host_available_mib,f"layer {index}")
        if count in {1,2,4,8,14}: checkpoints.append({"layers":count,"last_global_layer":index,"tensor_count":stage.loaded_tensor_count,"logical_bytes":stage.loaded_logical_bytes,"rss":proc(),"host":host(),"load_ms_since_embedding":round((time.perf_counter()-cumulative)*1000,3),"unique_shards":list(dict.fromkeys(loader.opened_shards))})
    before_buffer=proc(); stage.allocate_quant_buffer(); after_buffer=proc()
    if stage.quant_buffer_identity_count()!=1: raise RuntimeError("QuantBuffer is not shared")
    caches=stage.make_decode_caches(position=4096)
    if len(caches)!=14 or len({id(x) for x in caches.values()})!=14: raise RuntimeError("invalid independent caches")
    if sum(type(x).__name__=="GLACache" for x in caches.values())!=7: raise RuntimeError("invalid cache types")
    guard(a.max_rss_mib,a.min_host_available_mib,"buffer and caches")
    token=torch.tensor([[42]],dtype=torch.long)
    embedding_timing,hidden=measure(lambda:stage.embed(token))
    start=time.perf_counter()
    with torch.inference_mode(): output,new_caches,per_layer,diagnostics=stage.forward_layers(hidden,caches,True)
    validation_ms=(time.perf_counter()-start)*1000
    if output.shape!=(1,1,3584) or output.dtype!=torch.float32 or not torch.isfinite(output).all(): raise RuntimeError("invalid AtlasStage output")
    if any(x.next_position!=4097 for x in new_caches.values()): raise RuntimeError("cache position mismatch")
    benchmark,_=measure(lambda:stage.forward_layers(hidden,caches))
    start=time.perf_counter()
    with torch.inference_mode(): full,_c,_t,_d=stage.forward_layers(stage.embed(token),caches)
    full_ms=(time.perf_counter()-start)*1000
    guard(a.max_rss_mib,a.min_host_available_mib,"final")
    result={"environment":{"hostname":platform.node(),"python":platform.python_version(),"torch":torch.__version__,"threads":torch.get_num_threads(),"cuda_available":torch.cuda.is_available(),"limits":{"max_bytes":a.max_bytes,"max_rss_mib":a.max_rss_mib,"min_host_available_mib":a.min_host_available_mib}},"plan":{"tensor_count":len(atlas.tensors),"logical_bytes":atlas.nbytes},"initial":initial,"meta_rss":meta,"loading":{"checkpoints":checkpoints,"sampled_checksum":sampled,"embedding_sampled_checksum":embedding_checksum,"opened_shards":list(dict.fromkeys(loader.opened_shards)),"shard_open_events":len(loader.opened_shards)},"quant_buffer":{"identity_count":stage.quant_buffer_identity_count(),"bytes":stage.quant_buffer.nbytes,"before":before_buffer,"after":after_buffer},"caches":{"count":len(caches),"bytes":stage.cache_nbytes(caches),"types":{i:type(x).__name__ for i,x in caches.items()},"positions_before":sorted({x.next_position for x in caches.values()}),"positions_after":sorted({x.next_position for x in new_caches.values()})},"embedding":{"timing":embedding_timing,"shape":list(hidden.shape),"dtype":str(hidden.dtype),"finite":bool(torch.isfinite(hidden).all())},"validation":{"time_ms":round(validation_ms,3),"shape":list(output.shape),"dtype":str(output.dtype),"finite":bool(torch.isfinite(output).all()),"per_layer_ms":per_layer,"diagnostics":diagnostics,"mean_gla_ms":round(statistics.mean(v for i,v in per_layer.items() if i%2==0),3),"mean_attention_ms":round(statistics.mean(v for i,v in per_layer.items() if i%2),3)},"benchmark":benchmark,"complete_atlas":{"time_ms":round(full_ms,3),"shape":list(full.shape),"dtype":str(full.dtype),"finite":bool(torch.isfinite(full).all())},"final":{"process":proc(),"host":host()},"loader":{"tensor_count":stage.loaded_tensor_count,"logical_bytes":stage.loaded_logical_bytes,"materialized_bytes":loader.materialized_bytes}}
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
