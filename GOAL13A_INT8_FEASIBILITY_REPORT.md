# Goal 13A — True INT8 CPU feasibility audit

Date: 2026-08-16
Scope: audit only; no stable implementation, image, checkpoint, dependency, or distributed runtime was changed. No full-model experiment was run.

## Executive conclusion

A persistent low-precision weight path is technically realistic enough to justify one isolated real-layer benchmark, but no existing enabled PyTorch quantized `Linear` can be dropped into the stable runtime.

The most credible experiment is a small, standalone CPU kernel for decode (`M=1`) which consumes persistent signed weight codes in 128-element groups, accumulates integer products in `int32`, applies the checkpoint's FP32 group scales and per-token spike threshold, and optionally applies a sparse correction for weight codes outside INT8. This is plausible on both machines: Haswell can use AVX2/FMA, while Ivy Bridge can use SSE4.1/AVX for packing/scaling and SSE integer multiply/add instructions. Ivy Bridge has no fast dot-product instruction, so speedup is uncertain and must be measured rather than inferred. The principal potential wins there are eliminating per-forward divide/round/multiply and reducing weight traffic approximately fourfold.

This is a benchmark recommendation, not a recommendation to alter the stable distributed implementation.

## Evidence and method

### Directly observed facts

- Repository implementation inspected: `src/spikingbrain_cpu/ops.py`, `block.py`, `selective_loader.py`, their tests, stage scripts, Dockerfiles, and Goal 8.5/8.6 logs.
- Stable image inspected without modification: `spikingbrain-cpu:goal8.6-openblas`, image `sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836`.
- The image reports PyTorch `2.13.0+openblas.ivybridge`, `USE_MKL=0`, `USE_MKLDNN=0`, `USE_FBGEMM=0`, `USE_QNNPACK=0`, `USE_XNNPACK=0`, and no supported quantized engines (`torch.backends.quantized.supported_engines == []`, engine `none`).
- PyTorch source was checked at the exact build commit, `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Upstream W8ASpike source was checked at BICLab repository commit `ef99987167cf7386ab3348312c8c7aa00a6696ee`, especially `W8ASpike/quant_linear.py`, `modeling_gla_swa.py`, `gla_attention.py`, `window_attention.py`, and `Int2Spike/neuron.py`.
- The local read-only atlas subset's `config.json`, Safetensors index, and two real layer-0 weight/scale pairs were inspected. No forward or timing experiment was performed.

### Assumptions and items not established

- The upstream Git commit is treated as the intended public implementation. The local checkpoint does not contain its remote Python files, so exact provenance between that commit and the downloaded checkpoint is not cryptographically established.
- The learned scales appear intended to encode “W8”, but neither upstream `Quantizer.forward` nor the repository path clamps weight codes to INT8. Real checkpoint inspection proves some rounded codes exceed INT8; the training/export procedure that produced the scales was not present locally.
- No representative hidden-state corpus was captured in this audit. Activation spike-code ranges and saturation frequency must therefore be measured in Goal 13B.
- A custom SSE/AVX kernel is plausible, not proven faster. In particular, Ivy Bridge performance cannot be predicted from Haswell results.

## 1. Current `QuantLinear` end to end

For input `x`, weight `W[out,in]`, group size `G=128`, stored scales `s[out,in/G,1]`, and default `k=3`:

1. Activation threshold, per input row/token:

   `vth = clamp(mean(abs(x), dim=-1, keepdim=True, dtype=FP32) / 3, 1e-5, 1e4)`

2. Spike integer values:

   `a_code = round(x / vth)`

   There is no clamp and the tensor remains floating point in the current CPU implementation. Upstream optionally encodes and immediately decodes these integers through a bidirectional bitwise spike node; this does not make the eventual dense linear an INT8 operation.

3. Activation fake dequantization:

   `x_fq = (a_code * vth).to(x.dtype)`

4. Weight grouping and fake quantization:

   `w_code[o,g,k] = round(W[o,g,k] / s[o,g,0])`

   `W_fq[o,g,k] = w_code[o,g,k] * s[o,g,0]`

   This is symmetric, zero-centered quantization (`zeros` is registered as `None`). It is per output channel **and per consecutive 128-value K group**, not ordinary per-output-channel quantization. There is no zero point and, critically, no clamp in either upstream or local code.

5. Dense operation:

   `output = FP32_linear(x_fq, W_fq, bias)`

Dequantization occurs explicitly before the GEMM: activations are multiplied by `vth`, and weights by group scales, producing dense FP32 operands. OpenBLAS then performs FP32 GEMM/GEMV.

### Repeated work on every forward

- activation absolute value, reduction, division, clamp, division by threshold, rounding, and multiplication by threshold;
- for every weight element: FP32 division by its group scale, rounding, and FP32 multiplication by the scale;
- writing a full dense FP32 fake-quantized weight buffer;
- reading that buffer again in FP32 GEMM.

The repository reuses one maximum-sized FP32 buffer, avoiding repeated allocation and some transient tensors, but it does not cache quantized weights. The original FP32 weights and the equally large FP32 work buffer coexist.

For a one-token decode projection, weight preparation is expected to dominate or at least rival GEMV: it performs three elementwise operations over the full matrix and writes it before GEMV rereads it. Existing stable-runtime data support this diagnosis: on argo3 an MLP gate `QuantLinear` took 166.762 ms (GLA layer) or 182.306 ms (attention layer), and complete MLPs took 370–393 ms. These timings include both fake quantization and linear work; this audit did not re-time their components.

### Important MLP exception

`gate_proj` and `up_proj` call `QuantLinear.forward`. Upstream `GLU.forward` passes `down_proj.weight` directly into `swiglu_linear`, so `down_proj` bypasses its `QuantLinear.forward` and its stored scales. The local CPU implementation mirrors this. A future conversion must not assume all declared `QuantLinear` modules currently have the same runtime semantics.

## 2. Model and checkpoint semantics

The local configuration specifies 28 layers, hidden size 3584, intermediate size 18944, 28 attention heads, 4 KV heads, and head dimension 128. Relevant matrix shapes are:

| Projection | Weight shape `[out,in]` | 128-value groups per output |
|---|---:|---:|
| q/o | 3584 × 3584 | 28 |
| k/v/gk | 512 × 3584 | 28 |
| gate/up | 18944 × 3584 | 28 |
| down | 3584 × 18944 | 148 |

All indexed checkpoint tensors are accounted for as FP32; total declared size is 15,384,983,040 bytes. Each quantized projection has FP32 `weight` and FP32 `weight_quantizer.scales` tensors. There are no persistent INT8 weight tensors or zero points.

Real layer-0 inspection produced:

| Matrix | FP32 weight | Scale shape/bytes | Rounded code range | Codes outside `[-128,127]` |
|---|---:|---:|---:|---:|
| `attn.q_proj` 3584×3584 | 51,380,224 B | 3584×28×1 / 401,408 B | -182…194 | 107,934 / 12,845,056 (0.840%) |
| `mlp.gate_proj` 18944×3584 | 271,581,184 B | 18944×28×1 / 2,121,728 B | -173…173 | 705,638 / 67,895,296 (1.039%) |

Consequences:

- `round(W/s).to(int8)` is wrong because conversion would wrap, not preserve the current values.
- Saturating to INT8 is a semantic approximation whose output error must be measured.
- Exact persistent representation remains possible as clamped INT8 base values plus sparse `(position, residual)` outliers, or as INT16 codes. The sparse correction retains an INT8 main matrix and exact checkpoint codes, at additional compute and metadata cost.
- Activation codes are also unclamped. A true INT8×INT8 kernel must first establish their real range. An exact fallback for out-of-range activations or a sparse correction is required; unconditional saturation changes W8ASpike semantics.

Attention K/V `quantize_sym` is a separate per-vector symmetric 8-bit fake quantizer with scale `amax(abs(x))/127` and clamp `[-128,127]`. Query uses the dynamic spike rule. Those attention operations are outside the proposed one-layer benchmark.

## 3. CPU backend inventory

| Candidate | atlas5 / Haswell | argo3 / Ivy Bridge | Current runtime | Scale/shape fit | Finding |
|---|---|---|---|---|---|
| Current FP32 OpenBLAS | Works, AVX2/FMA host | Works, Ivy-target OpenBLAS | Enabled and stable | Exact after FP32 fake dequant | Baseline only; no persistent INT8 or INT8 GEMM. |
| PyTorch eager quantized `Linear` (`x86`/FBGEMM) | Hardware could run FBGEMM | **Cannot:** FBGEMM requires AVX2 or newer | Engines list is empty; prepack raises `NoQEngine` | Standard weight quantization is per tensor/channel, not K-group 128; activations ordinarily affine INT8 | Not usable in the stable image and not a two-host solution. FBGEMM's official build requirements state AVX2+. |
| PyTorch oneDNN/MKLDNN quantization | Potentially supports Haswell | oneDNN supports Ivy Bridge generally, but INT8 speed/path would require direct validation | Explicitly compiled out; reintroducing it would change the validated runtime | Primitives can express richer scales in some configurations, but not a drop-in match to unclamped spike codes | Not a Goal 13B first choice. It would require a separate derived runtime and renew ISA/stability validation after the prior MKL incident. |
| PyTorch `aten::_int_mm` | Registered | Registered | Tiny correctness call works, but source decides fast oneDNN path only for AVX512-VNNI | INT8/UINT8 × INT8 → INT32; group scales must be applied through 28 split-K calls/reduction | The source fallback is a scalar C++ triple loop when AVX512-VNNI oneDNN is unavailable. Thus it is not an accelerated Ivy/Haswell backend here and 28 calls are unattractive. Useful only as a correctness oracle. |
| PyTorch `aten::_weight_int8pack_mm` | AVX2-specialized path exists | Falls to the generic non-vectorized implementation | Registered despite no quantized engine | Takes floating activation, INT8 weights, and only one scale per output row | It is weight-only mixed FP/INT8 rather than integer GEMM, cannot express 128-K group scales in one call, and its Ivy path is scalar. Not the best common backend. |
| QNNPACK | Primarily ARM/mobile | Not a realistic x86 solution | Compiled out | Does not solve groupwise/unclamped semantics | Reject. |
| XNNPACK | Has x86 microkernels | Some x86 SSE-family kernels are plausible | Compiled out, so requires separate build/integration | Modern QC8 weight APIs are channel-oriented; no demonstrated exact 128-K group mapping | Possible comparison for an approximate weight-only path, but not the primary semantic-preserving candidate. Must not be represented as available now. |
| Standalone oneDNN | Haswell kernels plausible | Library lists Ivy Bridge support, but without VNNI/AVX2 INT8 may be weak | New dependency outside stable image | Could implement groups by multiple primitives or custom postprocessing | Scientifically possible, operationally larger and less direct than a small custom kernel. Secondary only if the custom kernel fails. |
| Custom C++ SSE4.1/AVX + AVX2 kernels | AVX2 integer lanes plus FMA scaling | SSE4.1 integer multiply/add; AVX only helps FP scaling, no AVX2/VNNI integer dot | Can be built only in an isolated benchmark image/extension | Can implement exact group-128 scales, unclamped-code handling, M=1, and inference-only layout | Best common feasibility candidate. Performance is unknown, especially on Ivy, but the avoided FP32 preparation and 4× base-weight compression make one-layer measurement justified. |
| llama.cpp/ggml kernels | Efficient quantized kernels exist | Several legacy-x86 paths exist | Not integrated | Existing block formats/scales do not directly match this checkpoint | SpikingBrain's hybrid GLA/SWA architecture is not demonstrated as supported. It is only a kernel/port comparison, not an RPC or implementation shortcut. |

Primary sources: [BICLab SpikingBrain repository](https://github.com/BICLab/SpikingBrain-7B), [FBGEMM hardware requirements](https://docs.pytorch.org/FBGEMM/fbgemm/development/BuildInstructions.html), [PyTorch x86 quantization backend description](https://pytorch.org/blog/celebrate-pytorch-2.0/), and [oneDNN supported hardware overview](https://github.com/uxlfoundation/oneDNN).

### What actually happens on argo3

Argo3 has AVX/F16C but no AVX2, FMA, AVX-512, or VNNI. Therefore:

- FBGEMM is not merely slower; its documented minimum is AVX2, so it is ineligible.
- The stable PyTorch build contains no quantized engine at all.
- `_int_mm` does not provide a hidden optimized legacy-x86 path: at the inspected PyTorch commit, absent AVX512-VNNI plus enabled oneDNN, it executes a scalar multiply/add loop parallelized across output elements.
- `_weight_int8pack_mm` has explicit AVX2 and AVX512 code, but its generic x86 implementation is scalar and its single output-channel scale is semantically insufficient.
- A custom Ivy kernel must deliberately target SSE4.1-era integer operations. It cannot claim VNNI-like throughput. For signed×signed values, it should widen bytes and use 16-bit multiply/add or another overflow-safe construction; naïve `pmaddubsw` can saturate intermediate 16-bit sums and is not generally exact.
- At decode `M=1`, the workload is largely a stream through weights. Reducing the gate matrix from 259.0 MiB FP32 to 64.75 MiB INT8 base storage and eliminating reconstruction is valuable enough to test even without a dot-product ISA. Whether instruction overhead and group/outlier corrections consume that gain is the central Goal 13B question.

## 4. Smallest scientifically useful benchmark

Use exactly one real matrix: `model.layers.0.mlp.gate_proj` (`18944 × 3584`, no bias) plus its real scale tensor from the read-only atlas checkpoint subset. It is preferred over a toy or the smaller q projection because it is a measured decode bottleneck, exercises the production K=3584/group=128 layout, contains real outliers, and yields a decisive memory-bandwidth test. It is still only one projection, not a layer block or model run.

For this matrix:

- current FP32 weight: 271,581,184 B = 259.00 MiB;
- current reusable FP32 fake-quant buffer: another 259.00 MiB;
- FP32 scales: 2,121,728 B = 2.02 MiB;
- persistent INT8 base: 67,895,296 B = 64.75 MiB;
- an exact sparse outlier representation adds implementation-dependent indices/residuals for 705,638 entries (1.039%); its actual allocated bytes must be reported, not estimated away.

## 5. Exact Goal 13B benchmark plan

Do not modify or import the optimized path into the stable distributed code. Create a standalone benchmark artifact in a derived image or isolated build directory based on the exact stable image, with the checkpoint mounted read-only. Retain the original image unchanged.

1. **Preflight and provenance**
   - Verify image ID, PyTorch build configuration, CPU flags, tensor names/shapes/dtypes, checkpoint hashes, OpenMP/thread count, affinity, and no swap pressure.
   - Run separately on atlas5 and argo3. Pin physical cores consistently; report 1-thread and host-appropriate thread count (atlas 4, argo 2 physical cores, with 4-thread SMT as an explicitly separate result if desired).

2. **Inputs**
   - Use decode shape `[1,3584]` as the primary case.
   - Obtain at least 32 representative real gate-projection inputs by a read-only, narrowly instrumented single-layer/input-capture procedure, not a full-model timing experiment. If authorization remains limited to no model execution, use saved hidden tensors already produced by prior runs; otherwise Goal 13B must explicitly authorize this capture. Also retain one fixed seeded synthetic vector for reproducibility.
   - Record `a_code` min/max, percentiles, and counts outside INT8 before choosing an activation policy.

3. **One-time preparation (timed separately)**
   - Load FP32 `W` and FP32 scales.
   - Compute rounded codes exactly as current code does.
   - Build (B1) saturated persistent INT8 and (B2) persistent INT8 plus sparse outlier residuals sufficient to recover every original rounded code.
   - Pack/transcode into the kernel's read-only layout. Measure wall time, peak RSS, final resident/logical bytes, and serialization size. Preparation is excluded from steady-state latency but reported and amortized over 1, 10, 100, and 1000 tokens.

4. **Paths to compare**
   - **A — current:** unchanged repository `quant_linear` using FP32 weight, real scales, reusable FP32 buffer, and current activation fake quantization.
   - **B1 — true INT8 approximate:** persistent saturated INT8 weights; INT8 spike activations when in range; custom group-128 INT32 accumulation; FP32 multiplication by `vth` and group scale; no dense FP32 weight materialization.
   - **B2 — true INT8 exact-code candidate:** B1 plus sparse weight outlier correction and exact activation fallback/correction. This must reproduce the mathematical rounded codes, subject only to floating reduction-order differences.
   - **C — dequantized reference:** prepare `W_fq` once in FP32, then run only current activation fake quantization plus FP32 `F.linear`. This separates the benefit from caching preparation from the benefit of integer storage/compute.
   - Use PyTorch `_int_mm` only as a small correctness oracle, not as the expected accelerated B implementation.

5. **Timing protocol**
   - Time activation preparation, kernel/GEMM, scale/outlier postprocessing, and total forward separately.
   - Warm pages and kernels before steady-state measurement; also report one cold run separately.
   - Use at least 10 warmups and enough measured iterations for at least 2 seconds per path, minimum 30; alternate randomized path order to reduce thermal/order bias.
   - Use monotonic wall time, report median, p10, p90, minimum, and bootstrap 95% confidence interval. Confirm output consumption so calls cannot be elided.

6. **Memory protocol**
   - Report tensor logical bytes by component and process RSS/HWM after load, after preparation, and during each path.
   - For A include original FP32 weight plus reusable buffer; for B include packed INT8, scales, outlier structures, scratch, and any retained FP32 source. Show a deployment-like B measurement after releasing the FP32 source.

7. **Numerical protocol**
   - Treat A output as the required behavioral reference and compute max absolute error, mean absolute error, RMSE, relative L2 error, cosine similarity, and top-error locations for every input.
   - C should isolate only FP32 reduction/order effects and should be very close to A.
   - Compare B1 saturation error separately from B2. Never allow wrapping casts.
   - Require B2's decoded integer codes to equal the current rounded codes exactly before accepting output comparisons.

8. **Decision rule after Goal 13B**
   - A path is worth integration study only if it is stable and finite on both hosts, B2 preserves codes exactly (or B1's approximation is explicitly accepted based on measured error), deployment-like memory is materially smaller, and steady-state total latency improves on **argo3** as well as atlas5.
   - Suggested practical threshold: median total forward at least 1.20× faster than A on each host with non-overlapping bootstrap confidence intervals. If argo3 fails this threshold, do not propose a cluster-wide INT8 implementation merely because atlas5 wins.

## Recommendation

GO-INT8-BENCH
