# SpikingBrain CPU Cluster

*Experimental CPU-only distributed inference for
SpikingBrain-7B-W8ASpike on legacy x86 hardware*

## Overview

SpikingBrain CPU Cluster is an experimental, documented implementation of a
two-node CPU inference path for the SpikingBrain-7B-W8ASpike checkpoint. It
replaces unavailable GPU-oriented operations with reference PyTorch CPU code,
loads only the tensors needed by each stage, and connects the stages with a
persistent Gloo process group over a private Ethernet link.

This is an independent community project, not an official or
upstream-supported SpikingBrain CPU implementation. Model weights and tokenizer
files are not included. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### AI-assisted development

This project was developed by Sergio (Uraroga) with substantial assistance
from ChatGPT and OpenAI Codex. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)
for the full development disclosure.

## Status

- **Experimental** code, tested on two specific legacy x86 machines.
- CPU-only two-node distributed inference.
- Working deterministic greedy text generation.
- Working local tokenizer, upstream chat template, and EOS handling.
- Working persistent, multi-turn terminal chat across both nodes.
- Stable custom PyTorch/OpenBLAS runtime for the tested Ivy Bridge host.
- Not performance optimized and not intended as a quality benchmark.

## What was achieved

- Pure PyTorch CPU fallback operators; no CUDA, Triton, FlashAttention, `fla`
  runtime, or custom GPU kernels.
- Selective, read-only Safetensors loading without materializing the whole
  checkpoint on either host.
- A measured 14/14 split across 28 layers.
- A persistent Gloo pipeline carrying FP32 hidden states in one direction and
  one `int64` token in the other.
- Persistent GLA and KV autoregressive caches across decode steps.
- Real completion generation, application of the real tokenizer chat template,
  and EOS termination.
- Persistent multi-turn chat with one model-stage load per process, explicit
  context-limit handling, `/reset`, and clean `/quit`/interrupt shutdown.
- A legacy-x86-safe runtime built with OpenBLAS and an Ivy Bridge compile target.

## Hardware used

| host | CPU | cores/threads | relevant ISA | RAM | role |
|---|---|---:|---|---:|---|
| atlas5 | Intel Core i5-4590 | 4C/4T | AVX2, FMA | ~31 GiB | rank 0 |
| argo3 | Intel Core i3-3240 | 2C/4T | Ivy Bridge, AVX, F16C; **no AVX2** | ~31 GiB | rank 1 |

The hosts were connected through private Gigabit Ethernet. The reported setup
used Linux and Docker with host networking on interface `eno1`.

## Model

Upstream project: `BICLab/SpikingBrain-7B`
Checkpoint tested: `Abel2076/SpikingBrain-7B-W8ASpike`

The inspected checkpoint contains **7,692,495,104 parameters** (~7.69B). Its
tensors are stored as FP32, corresponding to approximately **28.66 GiB** of
theoretical parameter storage.

The `W8ASpike` name does **not** mean that this checkpoint is stored as a true
INT8 weight checkpoint. In the implementation tested here, `QuantLinear`
receives FP32 weights and performs fake quantization at runtime. It is not an
INT8-storage/INT8-GEMM CPU backend.

## Distributed architecture

```text
prompt / tokenizer
        |
        v
atlas5 / rank 0
embedding
layers 0..13
        |
        | hidden [1, seq, 3584] FP32
        v
Gloo / Gigabit Ethernet
        |
        v
argo3 / rank 1
layers 14..27
norm
lm_head
argmax
        |
        | int64 token
        v
atlas5
```

At each decode step, atlas5 embeds the returned token and advances layers
0–13; argo3 advances layers 14–27 and computes the next greedy token. The
boundary tensor for one decode token is only 14,336 bytes, and measured Gloo
transfers were sub-millisecond in the stable-generation campaign. Compute,
especially fake `QuantLinear`, dominated the network cost.

## Why 14/14

The initial design proposed a 16/12 split to give more layers to the faster
AVX2 machine. Full synthetic block measurements changed that decision. The
14/14 estimate was about 5.329 s versus 5.036 s per stage for decode, while
16/12 was about 6.090 s versus 4.317 s. Prefill was also closer to balanced at
14/14 (5.807 s versus 6.293 s) than at 16/12 (6.637 s versus 5.394 s). The
balanced split also left more RAM headroom on atlas5. Goal 3 therefore replaced
the initial plan with the final 14/14 baseline, later confirmed with real
weights.

## CPU implementation

The reference path implements:

- RMSNorm;
- SwiGLU feed-forward blocks;
- sliding-window causal grouped-query attention;
- recurrent gated linear attention (GLA);
- dynamic spike activation;
- fake-quantized `QuantLinear`;
- recurrent GLA cache state;
- sliding KV caches with explicit positions.

These are CPU/reference implementations designed for correctness checks and
experimentation. They are not highly tuned CPU kernels and are not claimed to
be bit-equivalent to upstream GPU kernels on every input or runtime.

## SIGILL / MKL case study

The first real distributed generation completed successfully. A second,
process-independent session then terminated on argo3 with exit code 132. The
kernel recorded an invalid-opcode trap in `libtorch_cpu.so`.

Diagnostic runs reproduced the same offset. Symbolization identified
`mkl_vml_kernel_sExp_Z0HAynn+0xab`; disassembly showed an EVEX/ZMM AVX-512
instruction. The Ivy Bridge i3-3240 supports AVX/F16C but not AVX-512.
Environment mitigations involving `ATEN_CPU_CAPABILITY`, oneDNN/DNNL ISA
settings, and MKL ISA settings did not make that tested runtime stable.

The resolution in this environment was a custom PyTorch build with MKL and
MKLDNN disabled, OpenBLAS enabled, and an Ivy Bridge compile target. The
subsequent campaign completed independent stress processes, real stages,
distributed forwards, and text generations with zero new SIGILL events.

This is a case study of what was observed with this particular PyTorch build,
embedded MKL VML path, CPU, and environment. It is **not** a general claim that
MKL is broken.

## Stable runtime

Validated image tag and ID:

```text
spikingbrain-cpu:goal8.6-openblas
sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836
```

The runtime reports `PyTorch 2.13.0+openblas.ivybridge`. Principal build
settings are:

```text
USE_CUDA=0
USE_MKL=0
USE_MKLDNN=0
BLAS=OpenBLAS
USE_DISTRIBUTED=1
USE_GLOO=1
USE_MPI=0
USE_NCCL=0
CFLAGS=-O2 -march=ivybridge -mtune=ivybridge
CXXFLAGS=-O2 -march=ivybridge -mtune=ivybridge
```

`-march=native` was intentionally avoided. Building on the Haswell atlas5 host
with native flags could emit instructions unavailable on argo3; Ivy Bridge is
the compatibility floor shared by the tested machines.

## Reproducibility results

Goal 8R launched five complete sessions with new Docker processes, process
groups, model stages, and empty caches. For prompt `Hello`, all five produced:

```text
prompt ids:    [9707]
generated ids: [11, 323, 10565]
text:          Hello, and welcome
```

The campaign recorded 5/5 completions, zero SIGILL, zero process swap,
coherent positions across all 28 caches, and clean shutdown after every
session. Median session time was **13.223680 s/token**. The first session was a
cold-start outlier; raw and aggregate values are preserved in
[GOAL8R_STABLE_GENERATION.md](GOAL8R_STABLE_GENERATION.md) and
[goal8r_generation_summary.json](goal8r_generation_summary.json).

## Real prompt examples

The Goal 9 smoke tests produced exactly:

```text
The capital of France is Paris. The capital of
Il cielo è un'immagine di
Hello, how are you? I am fine, thank
```

The real tokenizer chat template produced this continuation and then EOS:

```text
Reply briefly: 4.<|endoftext|>
```

These are short technical smoke tests, **not** a formal or representative
model-quality benchmark.

## Performance

| campaign/path | reported time | scope |
|---|---:|---|
| Goal 7, old MKL runtime | ~10.684 s | technical distributed forward |
| Goal 8R, stable OpenBLAS | ~13.224 s/token | median greedy generation session rate |
| Goal 9, real prompts | ~14 s/token | short-prompt greedy generation |

These numbers are not directly equivalent because they measure different
paths and aggregation methods. They are included for engineering context, not
as a performance comparison. Runtime stability was preferred over speed.

## Main bottleneck

`QuantLinear` performs fake quantization on FP32 weights during every forward.
It does not store weights as true INT8 or call an INT8 GEMM backend. This
repeated work dominates inference and is the main candidate for future
optimization, but it was deliberately left unchanged during runtime
stabilization and publication freeze.

## Persistent chat

Goal 14 adds a terminal chat loop while keeping both model stages loaded in
RAM. For every user turn, rank 0 renders the complete conversation with the
tokenizer's real chat template. Both ranks then create fresh caches and rebuild
the conversation state for that turn; model weights are not reloaded.

Use `/reset` to clear history and `/quit` to shut down both ranks. The launcher
requires the local/remote model and tokenizer paths:

```bash
ATLAS_MODEL=/path/to/atlas-stage \
TOKENIZER=/path/to/tokenizer \
ARGO_MODEL=/path/on/argo3/to/checkpoint \
scripts/run_chat_cluster.sh
```

See [CHAT_README.md](CHAT_README.md) for deployment variables and concise usage
instructions, and [GOAL14_PERSISTENT_CHAT.md](GOAL14_PERSISTENT_CHAT.md) for the
engineering report.

## Repository layout

```text
src/spikingbrain_cpu/   CPU operators, blocks, partitioning, loader, protocol
scripts/                diagnostics, stage tools, and distributed entry points
tests/                  operator, block, loader, partition, and protocol tests
CHAT_README.md          practical persistent-chat launcher guide
Dockerfile*             original CPU environment and stable OpenBLAS build
GOAL*.md                chronological engineering reports
goal*.json              selected aggregate, machine-readable summaries
```

Raw logs, per-rank outputs, model files, wheels, and Docker images are ignored.

## Requirements

- Docker with enough local storage to build or load the runtime image;
- two Linux x86-64 hosts with enough RAM for one partition each;
- a private or otherwise trusted LAN connection between the hosts;
- the model checkpoint and tokenizer downloaded separately under their own
  terms;
- the same project code and validated runtime on both nodes.

The documented machines each had approximately 31 GiB RAM. Other hardware and
BLAS/runtime combinations may behave differently.

## Quick start

There is intentionally no one-command installer. Read
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and verify upstream model terms first.
The following is a parameterized outline, not a copy-paste deployment script.

1. Build the stable image from `Dockerfile.safe-runtime`, or transfer one image
   built once to both hosts. Verify that both hosts report the same image ID.
2. Place the full checkpoint on argo3 and an atlas subset containing embedding
   plus layers 0–13 on atlas5. Keep both read-only. Place tokenizer files on
   atlas5 separately.
3. Confirm that `<ATLAS_IP>` and `<ARGO_IP>` are addresses on the intended
   private interface and that the same free port is used on both commands.
4. Start rank 1 on argo3 first:

```bash
docker run --rm --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -v <MODEL_PATH>:/model:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_argo.py \
    --master-addr <ATLAS_IP> --peer <ATLAS_IP> \
    --model-dir /model --max-bytes 16000000000 \
    --max-prompt-tokens 16 --max-new-tokens 5 --threads 4
```

5. Start rank 0 on atlas5 with the same port (default 29500 unless explicitly
   changed):

```bash
docker run --rm --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v <MODEL_PATH>:/model:ro \
  -v <TOKENIZER_PATH>:/tokenizer:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_atlas.py \
    --master-addr <ATLAS_IP> --peer <ARGO_IP> \
    --model-dir /model --tokenizer-dir /tokenizer \
    --prompt 'Hello' --max-prompt-tokens 16 --max-new-tokens 5 \
    --max-bytes 16000000000 --threads 4
```

Review both JSON reports, cache positions, swap fields, shutdown status, and
the argo3 kernel journal before treating a session as valid.

## Tests

No host PyTorch installation is required. With the stable image already
available:

```bash
docker run --rm spikingbrain-cpu:goal8.6-openblas python -m pytest -q
```

The publication freeze ran the package import and the relevant operator,
block, loader, partition, and protocol suite without loading real weights.

## Experimental history

The Goal reports are preserved as an engineering log, including failed
approaches and changed decisions:

- [Initial design / Goal 1](PROGETTO_CLUSTER_CPU.md)
- [Goal 2: CPU fallbacks](GOAL2_CPU_FALLBACKS.md)
- [Goal 2.5: Docker environment](GOAL2_5_DOCKER_BENCHMARK.md)
- [Goal 3: block benchmark and 14/14 split](GOAL3_BLOCK_BENCHMARK.md)
- [Goal 4: real-weight smoke test](GOAL4_REAL_WEIGHTS_SMOKE_TEST.md)
- [Goal 5: ArgoStage](GOAL5_ARGO_STAGE.md)
- [Goal 6: AtlasStage](GOAL6_ATLAS_STAGE.md)
- [Goal 7: Gloo end-to-end](GOAL7_GLOO_END_TO_END.md)
- [Goal 8: first generation and SIGILL](GOAL8_FIRST_GENERATION.md)
- [Goal 8.5: SIGILL diagnosis](GOAL8_5_SIGILL_DIAGNOSIS.md)
- [Goal 8.6: safe OpenBLAS runtime](GOAL8_6_SAFE_RUNTIME.md)
- [Goal 8R: stable generation](GOAL8R_STABLE_GENERATION.md)
- [Goal 9: real prompts](GOAL9_REAL_PROMPTS.md)
- [Goal 14: persistent two-node chat](GOAL14_PERSISTENT_CHAT.md)

See [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) for a compact chronology.

## Limitations

- Tested only on the two machines described above.
- The checkpoint is FP32 and requires substantial RAM.
- Generation is slow and latency-oriented; throughput was not optimized.
- There is no true INT8-storage/INT8-GEMM CPU backend.
- No formal model-quality evaluation was performed.
- Identical output is not guaranteed with a different BLAS, PyTorch build,
  operator implementation, or hardware platform.
- The code is experimental and not hardened as a production serving system.
- This is not an upstream-supported CPU implementation.

No priority claim is made. The repository documents an experimental
implementation and the measurements observed on one two-node setup.

## License and acknowledgements

The code and documentation in this repository are released under the MIT
License. Upstream models, repositories, and dependencies retain their own
licenses and terms. See
[LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md), including the AI-assisted
development disclosure.
