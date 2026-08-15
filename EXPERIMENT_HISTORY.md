# Experiment history

This chronology intentionally preserves failed attempts and revised decisions.
The original reports remain the source of detailed measurements.

## Goal 1 — Design and checkpoint inventory

**Objective:** determine whether a ~7.69B FP32 checkpoint could fit across two
~31 GiB CPU hosts and design a selective-loading pipeline.

**Result/decision:** **GO (conditional prototype)**. Header-only inspection
showed ~28.66 GiB of parameters and made a two-node split feasible. The initial
proposal was 16 layers on atlas5 and 12 on argo3, conditional on CPU fallback
and block benchmarks. No tensor data was loaded.

**Change:** established the architecture, memory budget, and initial 16/12
hypothesis. See [PROGETTO_CLUSTER_CPU.md](PROGETTO_CLUSTER_CPU.md).

## Goal 2 — CPU fallbacks

**Objective:** implement and validate CPU/reference replacements for the model's
GPU-oriented operations.

**Result/decision:** **GO-CPU-FALLBACKS**. Tests covered RMSNorm, SwiGLU,
attention, recurrent GLA, dynamic spikes, fake `QuantLinear`, and caches.

**Change:** created the core `src/spikingbrain_cpu` operator/block code and
documented that correctness tests did not imply GPU bit-equivalence or speed.

## Goal 2.5 — Docker-only environment

**Objective:** remove reliance on host Python and test one CPU Docker image on
both machines.

**Result/decision:** **GO-DOCKER**. Tests passed on both hosts, but argo3 showed
a material penalty in small/vectorized operations and fake `QuantLinear`.

**Change:** Docker became the required execution environment; the 16/12 split
was left open pending full-block measurements.

## Goal 3 — Full synthetic blocks and partition

**Objective:** benchmark complete GLA/attention blocks and estimate stage RAM
and time for candidate splits.

**Result/decision:** **GO-14/14**. Full-block results contradicted the initial
assumption that 16/12 would balance compute. The 14/14 split was closer in both
decode and prefill while preserving RAM headroom.

**Change:** final baseline became atlas layers 0–13 and argo layers 14–27.

## Goal 4 — Real-weight smoke test

**Objective:** validate selective loading and one real block without copying or
mutating checkpoint data.

**Result/decision:** **GO-REAL-WEIGHTS**. Shapes, loading plans, memory limits,
and real-weight forward behavior passed using read-only mounts.

**Change:** moved from synthetic tensors to controlled real checkpoint access.

## Goal 5 — ArgoStage

**Objective:** load and execute layers 14–27, final norm, and `lm_head` on
argo3 within memory limits.

**Result/decision:** **GO-ARGO-STAGE**. The complete second stage loaded and ran;
global cache indices and tensor accounting matched the 14/14 plan.

**Change:** established the real rank-1 stage.

## Goal 6 — AtlasStage

**Objective:** create a safe atlas subset and run embedding plus layers 0–13.

**Result/decision:** **GO-ATLAS-STAGE**. The first stage loaded, ran, and stayed
within RAM without process swap.

**Change:** completed both independently executable real-weight stages.

## Goal 7 — Distributed Gloo end-to-end

**Objective:** connect the two stages over private Ethernet with a persistent
CPU Gloo process group.

**Result/decision:** **GO-GLOO-END-TO-END**, with a non-reproduced SIGILL noted
for mandatory monitoring. Synthetic transfer was byte-identical and real
distributed forwards completed with clean shutdown.

**Change:** established the one-boundary protocol and end-to-end technical
forward path.

## Goal 8 — First real text generation

**Objective:** add tokenizer, greedy autoregressive feedback, and persistent
caches.

**Result/decision:** **STOP-SIGILL**. Session 1 generated
`Hello, and welcome`. A new process in session 2 exited 132; argo3 recorded an
invalid opcode in `libtorch_cpu.so`.

**Change:** generation worked functionally, but the runtime was declared unsafe
and further sessions stopped.

## Goal 8.5 — SIGILL diagnosis

**Objective:** reproduce and identify the illegal instruction without changing
model math.

**Result/decision:** **GO-RUNTIME** (replace the runtime before continuing).
Repeated traps shared one offset, symbolized as
`mkl_vml_kernel_sExp_Z0HAynn+0xab`; disassembly showed AVX-512/ZMM on an Ivy
Bridge CPU. ATen, DNNL, and MKL environment restrictions did not provide a
stable mitigation in this setup.

**Change:** selected a custom non-MKL runtime as the safe next step rather than
continuing generation.

## Goal 8.6 — Safe OpenBLAS runtime

**Objective:** build and stress a PyTorch runtime compatible with the oldest
CPU without changing model operators or partition.

**Result/decision:** **GO-RUNTIME-STABLE**. PyTorch 2.13.0 was built with
OpenBLAS, no MKL/MKLDNN, and an Ivy Bridge target. Independent microtests,
30 ArgoStage runs, AtlasStage runs, and five distributed technical sessions
completed with zero SIGILL.

**Change:** froze `spikingbrain-cpu:goal8.6-openblas` as the stable baseline.

## Goal 8R — Stable generation repetition

**Objective:** repeat real greedy generation in five completely new process
pairs using the safe runtime.

**Result/decision:** **GO-GENERATION-STABLE**. All 5/5 sessions generated
identical IDs `[11, 323, 10565]`, with zero SIGILL, zero swap, coherent caches,
and clean shutdown.

**Change:** demonstrated repeatability across process lifetimes and replaced
the failed Goal 8 runtime result with a stable runtime baseline, without hiding
the original failure.

## Goal 9 — Real prompts and chat template

**Objective:** run short English, Italian, conversational, and templated-chat
smoke tests with longer decode and EOS handling.

**Result/decision:** **GO-PROMPTS**. Three raw prompts and one real tokenizer
chat-template session completed with finite hidden/logits, zero swap, zero
kernel traps, coherent caches, and clean shutdown. The chat session emitted
real EOS. An initial chat attempt exposed an application guard limited to eight
prefill tokens; the guard was parameterized and the session restarted from
new processes.

**Change:** documented raw and chat behavior while explicitly avoiding a
general quality claim.
