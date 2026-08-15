# Reproducibility

This document separates facts measured in the completed experiments from the
interpretation drawn from them. Model files are not distributed here.

## Measured facts

### Hardware and operating environment

| property | atlas5 | argo3 |
|---|---|---|
| CPU | Intel Core i5-4590 @ 3.30 GHz | Intel Core i3-3240 @ 3.40 GHz |
| topology | 4 cores / 4 threads | 2 cores / 4 threads |
| relevant flags | AVX, F16C, AVX2, FMA | AVX, F16C; no AVX2 |
| RAM | approximately 31 GiB | approximately 31 GiB |
| OS at publication audit | Ubuntu 24.04.4 LTS | Ubuntu 24.04.4 LTS |
| kernel at publication audit | Linux 7.0.0-28-generic | Linux 7.0.0-28-generic |
| Docker server | 29.7.2 | 29.6.2 |

The hosts communicated over private Gigabit Ethernet using Linux interface
`eno1`, MTU 1500, Docker `--network host`, and Gloo. Experimental reports retain
the private IP addresses as historical measurements; public commands use
`<ATLAS_IP>` and `<ARGO_IP>` placeholders.

### Model and partition

- Upstream architecture attribution: BICLab / SpikingBrain-7B.
- Tested checkpoint: Abel2076/SpikingBrain-7B-W8ASpike, obtained separately.
- 7,692,495,104 inspected parameters (~7.69B).
- FP32 checkpoint tensor storage, approximately 28.656777 GiB theoretical.
- 28 transformer layers and hidden size 3584.
- atlas5/rank 0: embedding and global layers 0–13.
- argo3/rank 1: global layers 14–27, final norm, and `lm_head`.
- Checkpoint and tokenizer mounts were read-only.
- No model weights or tokenizer copy is part of this repository.

### Stable runtime

```text
image:    spikingbrain-cpu:goal8.6-openblas
image ID: sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836
PyTorch:  2.13.0+openblas.ivybridge
BLAS:     OpenBLAS
threads:  4 per rank during measured generation
```

Principal build settings:

```text
USE_CUDA=0
USE_MKL=0
USE_MKLDNN=0
BLAS=OpenBLAS
USE_DISTRIBUTED=1
USE_GLOO=1
USE_MPI=0
USE_NCCL=0
BUILD_TEST=0
CFLAGS=-O2 -march=ivybridge -mtune=ivybridge
CXXFLAGS=-O2 -march=ivybridge -mtune=ivybridge
```

The PyTorch source was pinned to tag `v2.13.0`, verified commit
`cf30153c4c131c8164ee7798e5022d810682e2cb`. The image was built once and
transferred to the second host; both nodes reported the same image ID.

### Tokenizer and generation

- Local `Qwen2Tokenizer` files from the separately obtained checkpoint source.
- Offline mode: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- No manually inserted BOS in raw completion tests.
- Greedy decoding: `next_token = argmax(logits)`.
- Sampling disabled; temperature not applicable.
- Goal 8R: prompt `Hello`, `max_new_tokens=3`.
- Goal 9 raw prompts: `max_new_tokens=5`, prompt limit 16.
- Goal 9 chat: tokenizer `apply_chat_template`, maximum 8 new tokens; real EOS
  occurred at token 7.

### Independent-session methodology

Each valid session used new Docker processes on both nodes, a new Gloo process
group, newly loaded stages, empty caches, identical thread settings, and the
same checkpoint/runtime. Multiple generations inside one process were not
counted as independent sessions. Rank 1 was started first and rank 0 joined it
on a dedicated port.

For Goal 8R, five such sessions generated identical IDs
`[11, 323, 10565]`. For Goal 9, each prompt used one independent session; the
chat session was independently restarted after an application-level prompt
length guard was corrected.

### Numerical and cache verification

The drivers checked that received and produced hidden states and logits were
finite and that returned token IDs were in range. Cache dictionaries were
checked against the expected 14 global layer indices on each rank. Every cache
had the prompt length as its post-prefill position and advanced by exactly one
for each generated token fed back into the model.

Observed cache storage grew by 28,672 bytes per processed decode token and per
rank. A final token that ended on `max_new_tokens`, and the final EOS token, was
not fed back through the model.

### SIGILL and swap verification

Before a campaign, the current timestamp was recorded on argo3. After each
session and at campaign end, `journalctl -k --since <timestamp>` was searched
for `SIGILL`, `invalid opcode`, and traps involving `libtorch_cpu.so`. A new
match was a stop condition. Goal 8R and Goal 9 both recorded zero new matches.

Process RSS, high-water mark, and `VmSwap` were read from `/proc/self/status`;
host `MemFree` and `MemAvailable` came from `/proc/meminfo`. `VmSwap` was zero
on both ranks in every valid Goal 8R and Goal 9 session.

### Recorded outcomes

- Goal 8R: 5/5 complete, identical greedy tokens, zero SIGILL, zero swap,
  coherent caches, clean shutdown; median 13.223680 s/token. Session 1 was a
  cold-start outlier.
- Goal 9: three raw completion prompts plus one real chat-template prompt;
  finite values, zero swap, zero kernel traps, clean shutdown, and real EOS in
  the chat session.

See [goal8r_generation_summary.json](goal8r_generation_summary.json) and
[goal9_prompts.json](goal9_prompts.json) for machine-readable values.

## Interpretation

- The 14/14 split was chosen because measured full-block time and estimated
  memory were more balanced than the initial 16/12 proposal.
- The private Ethernet boundary is not the primary latency bottleneck in this
  setup; the hidden transfer is small relative to CPU stage time.
- Runtime fake quantization of FP32 weights in `QuantLinear` is the principal
  observed optimization candidate.
- Disabling MKL/MKLDNN and targeting Ivy Bridge removed the illegal-instruction
  path in the subsequent tested campaign. This is evidence about this setup,
  not a universal conclusion about MKL or other CPUs.
- The short Goal 9 outputs are smoke tests, not evidence of general model
  quality.

Reproduction on different hardware, PyTorch commits, BLAS libraries, or model
files may yield different timings or greedy tokens.
