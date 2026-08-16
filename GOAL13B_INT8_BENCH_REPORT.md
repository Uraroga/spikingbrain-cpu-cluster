# Goal 13B — Isolated true-INT8 real-matrix benchmark

Date: 2026-08-16
Decision: the isolated exact-code INT8 path exceeded the Goal 13A latency and memory criteria on **both** hosts, including Ivy Bridge argo3.

## Scope and preservation

This experiment used only `model.layers.0.mlp.gate_proj`, shape `18944 × 3584`, and its real group scales. The checkpoint was mounted read-only. No full-model benchmark was run. The stable distributed implementation and stable image were not modified or retagged; both hosts retained stable image ID `sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836`.

The compiler toolchain was added only to a derived image, `spikingbrain-cpu:goal13b-bench`, ID `sha256:451ccf06388a54546a5a5611ae93a51c284c613cfb3c0392a41d3392d5ecdba4`. The benchmark code is isolated under `goal13b/`; it is not imported by the production stage code. No commit or push was performed.

## Experimental implementation

### Representative inputs

No previous hidden-state captures existed. A narrow, read-only capture was therefore made on atlas5:

- 32 real tokenizer IDs from a fixed English prompt;
- real checkpoint embedding;
- only layer 0's attention side;
- capture at the output of layer 0 `mlp_norm`, which is the actual input to `mlp.gate_proj`;
- shape `[32,3584]`, FP32, all finite;
- no remaining layers, MLP projection, LM head, generation, or full-model timing.

Each captured row was benchmarked as an independent decode-shaped `[1,3584]` input. A fixed seed-1313 synthetic vector was added for numerical validation only.

Across all 33 inputs, dynamic spike codes ranged from `-16` to `16`; zero activation codes fell outside INT8. This result applies to this capture, not to all possible prompts or layers.

### Paths

- **A — current:** the repository's unchanged `quant_linear`, including per-forward weight divide, round, multiply into a reusable FP32 buffer, then FP32 `F.linear`.
- **B1 — true INT8 approximate:** persistent checkpoint codes saturated to signed INT8; activation spike codes saturated to signed INT8; custom group-128 signed INT8 × signed INT8 accumulation to INT32; FP32 application of group scales and spike threshold.
- **B2 — true INT8 exact-code:** the same persistent INT8 base, exact unclamped activation codes in INT16, and a sparse `(K index, INT16 residual)` correction for out-of-range weight codes. The base dot products remain INT8-weight integer GEMM/GEMV. B2 reconstructed every current rounded checkpoint weight code exactly before timing.
- **C — dequantized reference:** `W_fq = round(W/s) × s` prepared once in FP32; current activation fake quantization followed by FP32 `F.linear`.

The custom kernel uses overflow-safe widening before 16-bit multiply/add. Atlas selected its AVX2 implementation. Argo selected its SSE4.1 implementation and never executed AVX2, FMA, AVX-512, or VNNI code. Scaling and sparse correction were measured separately.

## Provenance and controls

| Property | atlas5 | argo3 |
|---|---|---|
| CPU | Intel i5-4590, 4C/4T | Intel i3-3240, 2C/4T |
| ISA selected by B kernel | AVX2 | SSE4.1 (`avx2_kernel=false`) |
| Stable base runtime | PyTorch 2.13.0 OpenBLAS/Ivy target | identical |
| Host-appropriate run | CPUs 0–3, 4 threads | CPUs 0–1 (different physical cores), 2 threads |
| Additional run | CPU 0, 1 thread | CPU 0, 1 thread |
| Process swap | 0 MiB | 0 MiB |

Checkpoint config and index hashes were stable across repeated runs on each host. The atlas subset and argo full checkpoint used different index files and shard layouts but yielded identical matrix preparation statistics, packed SHA-256, numerical results, and B2 checksum.

Timing used 10 warmups, randomized alternating path order, a minimum of 30 observations and at least two measured seconds per path, monotonic wall time, consumed outputs, and 2,000-resample bootstrap confidence intervals for the median. Cold latency was recorded separately. The primary comparison is the host-appropriate thread count.

## Checkpoint preparation

The real matrix contained 67,895,296 codes with range `[-173,173]`. Exactly 705,638 codes (1.039%) were outside signed INT8.

| One-time operation | atlas5, 4 threads | argo3, 2 threads |
|---|---:|---:|
| Complete B2 preparation | 415.96 ms | 695.11 ms |
| Persistent logical bytes | 74,402,412 | 74,402,412 |
| Serialized bytes | 74,405,103 | 74,405,103 |
| Packed SHA-256 | `59295b22…02ef9e` | identical |

Persistent bytes comprise:

| Component | Bytes |
|---|---:|
| INT8 base matrix | 67,895,296 |
| FP32 group scales | 2,121,728 |
| outlier row pointers | 151,560 |
| outlier K indices | 2,822,552 |
| INT16 residuals | 1,411,276 |
| **persistent total** | **74,402,412** |

The B2 execution set additionally needs 2,121,728 bytes of INT32 group scratch and 7,168 bytes of INT16 activation, totaling 76,531,308 logical bytes. A's FP32 weight, scales, and reusable FP32 fake-quant buffer total 545,284,096 bytes. Thus B2's execution representation is 7.12× smaller than A's relevant representation (86.0% reduction).

A clean stable-image process loading and touching only the serialized B2 tensors increased RSS from 201.63 to 277.61 MiB on atlas and from 202.14 to 278.01 MiB on argo, approximately 76 MiB on each. In the preparation process, RSS did not return fully to the logical size after releasing FP32 sources because PyTorch's allocator retained memory; the clean-process result is the deployment-like measurement.

## Steady-state latency

### Primary host-appropriate result

| Host/path | n | Median ms | Bootstrap median 95% CI ms | p10–p90 ms | Minimum ms | Speedup vs A |
|---|---:|---:|---:|---:|---:|---:|
| atlas A, 4T | 30 | 212.079 | 197.327–223.900 | 162.096–246.400 | 157.990 | 1.00× |
| atlas B1, 4T | 161 | 9.257 | 8.095–11.983 | 4.403–23.182 | 4.347 | 22.91× |
| atlas B2, 4T | 114 | 15.221 | 10.964–17.206 | 5.581–33.318 | 5.265 | **13.93×** |
| atlas C, 4T | 34 | 59.588 | 57.727–62.503 | 50.642–72.205 | 48.592 | 3.56× |
| argo A, 2T | 30 | 145.468 | 145.318–145.633 | 145.066–145.824 | 144.794 | 1.00× |
| argo B1, 2T | 395 | 5.064 | 5.061–5.066 | 5.032–5.104 | 4.983 | 28.73× |
| argo B2, 2T | 309 | 6.468 | 6.465–6.472 | 6.418–6.520 | 6.360 | **22.49×** |
| argo C, 2T | 40 | 50.698 | 50.668–50.727 | 50.627–50.834 | 50.574 | 2.87× |

Argo is decisive: B2's upper CI bound, 6.472 ms, is far below A's lower bound, 145.318 ms. It exceeds the required 1.20× threshold by a wide margin while using only the Ivy-compatible SSE4.1 kernel.

Atlas's four-thread results were substantially noisier than argo's and its one-thread results; nevertheless B2's upper CI bound, 17.206 ms, remains far below A's lower bound, 197.327 ms.

### One-thread control

| Host/path | A ms | B1 ms | B2 ms | C ms | A/B2 |
|---|---:|---:|---:|---:|---:|
| atlas, AVX2 | 171.393 | 7.165 | 8.858 | 54.269 | 19.35× |
| argo, SSE4.1 | 182.612 | 7.877 | 10.029 | 51.418 | 18.21× |

The one-thread control confirms that the result is not created by thread-count asymmetry. It also demonstrates that argo's legacy SSE path is genuinely accelerated relative to current A, rather than falling into PyTorch's scalar `_int_mm` fallback.

### B component medians, host-appropriate runs

| Component | atlas B1 ms | atlas B2 ms | argo B1 ms | argo B2 ms |
|---|---:|---:|---:|---:|
| activation preparation | 0.162 | 0.166 | 0.216 | 0.201 |
| base integer kernel | 6.696 | 7.478 | 3.510 | 3.505 |
| sparse outlier correction | — | 0.930 | — | 0.897 |
| FP32 group scaling/reduction | 1.113 | 1.426 | 1.319 | 1.843 |

Component medians need not sum exactly to total medians because path order was randomized and atlas exhibited scheduling variance.

Cold latencies were: atlas A/B1/B2/C = 172.25/4.49/6.20/50.74 ms; argo = 181.79/5.22/6.45/50.82 ms. Cold B paths did not show a material packing penalty because packing was deliberately accounted as one-time preparation.

## Preparation amortization

Using complete B2 preparation plus host-appropriate B2 median forward:

| Host | 1 token | 10 tokens | 100 tokens | 1000 tokens | Steady B2 |
|---|---:|---:|---:|---:|---:|
| atlas | 431.18 ms/token | 56.82 | 19.38 | 15.64 | 15.22 |
| argo | 701.58 ms/token | 75.98 | 13.42 | 7.16 | 6.47 |

Preparation breaks even against A after approximately 3 tokens on atlas and 6 tokens on argo. Persisting the packed representation removes preparation from normal startup except for deserialization.

## Numerical results

A was the behavioral reference. Metrics below are worst case across 32 real inputs plus the fixed synthetic input, except where marked median.

| Path | Max abs | Worst mean abs | Worst RMSE | Worst relative L2 | Median relative L2 | Minimum cosine |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 3.2344e-2 | 1.7738e-3 | 2.8360e-3 | 3.2497e-3 | 7.1808e-4 | 0.9999948521 |
| B2 | 2.8610e-6 | 1.7630e-7 | 3.4194e-7 | 5.0576e-7 | 1.2552e-7 | 0.9999999999999 |
| C | 0 | 0 | 0 | 0 | 0 | effectively 1.0 |

B1's error is solely an approximation decision for the 1.039% saturated weight codes in this sample; activation saturation did nothing because all observed activation codes fit INT8. B2 recovered all weight codes exactly. Its tiny output difference from A comes from groupwise INT32 accumulation followed by FP32 scaling/reduction rather than OpenBLAS's FP32 product/reduction order. C was bit-identical to A for all tested inputs.

## Interpretation and limitations

The measured speedup is larger than a pure INT8 arithmetic estimate because A is not just FP32 GEMV: it reconstructs all 67.9 million fake-quantized FP32 weights on every call. C shows that caching those weights alone provides 2.87× on argo and 3.56× on atlas. B2 then gains further from the approximately fourfold smaller streamed base matrix and a decode-specific integer kernel.

These results establish feasibility, not production readiness:

- only one real layer-0 gate matrix was measured;
- the 32 real inputs came from one prompt and one layer; other layers/prompts may contain activation outliers, requiring the already-defined B2 exact activation representation or a wider fallback;
- B2 currently emits a large `[out,groups]` INT32 scratch tensor and performs scaling in PyTorch; fusion should reduce overhead but must be revalidated;
- the kernel and packed format need tests for arbitrary shapes, thread safety, serialization/versioning, errors, and unsupported CPUs;
- integration must preserve the `down_proj` bypass behavior and separately evaluate every projection family;
- stage-level benefits will be smaller than this projection-level speedup because norms, attention/GLA, `lm_head`, loading, and communication remain unchanged;
- atlas four-thread jitter should be investigated under a longer controlled campaign before relying on fine-grained scaling conclusions.

## Decision against Goal 13A rule

| Criterion | Result |
|---|---|
| Stable and finite on both hosts | Pass |
| B2 rounded weight codes exact | Pass |
| Numerical agreement with A | Pass; worst relative L2 5.06e-7 |
| Deployment memory materially smaller | Pass; 7.12× smaller relevant execution representation |
| Atlas median improvement ≥1.20× with non-overlapping CI | Pass; 13.93× |
| **Argo median improvement ≥1.20× with non-overlapping CI** | **Pass; 22.49× using SSE4.1** |

The next phase should be an integration **study**, not an immediate stable-runtime replacement: formalize the packed format and kernel tests, capture broader layer/input distributions, benchmark all projection shapes on both ISAs, and then measure one isolated block before considering the distributed path.

## Recommendation

GO-INT8-INTEGRATION-STUDY
