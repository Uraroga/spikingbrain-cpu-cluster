# Draft upstream issue — do not publish yet

## Proposed title

CPU-only distributed inference for SpikingBrain-7B-W8ASpike on two legacy x86 hosts

## Proposed body

Hello, and thank you for publishing SpikingBrain.

I have been working on an independent experimental CPU-only inference path for
the SpikingBrain-7B-W8ASpike checkpoint. The current prototype uses pure
PyTorch reference fallbacks for the model-specific operations, selective
Safetensors loading, and a 14/14 split across two ~31 GiB x86 hosts connected
with a persistent Gloo process group.

The prototype now completes greedy autoregressive generation, maintains GLA
and KV caches, applies the checkpoint tokenizer's chat template, and handles
EOS. Five completely independent stable-runtime sessions for `Hello` produced
identical greedy output, with zero SIGILL, zero process swap, coherent cache
positions across all 28 layers, and clean shutdowns. Short raw/chat smoke tests
also completed.

On an Ivy Bridge i3-3240, the tested PyTorch/MKL runtime intermittently reached
an AVX-512/ZMM instruction in `mkl_vml_kernel_sExp_Z0HAynn`, causing SIGILL.
The stable resolution was an Ivy Bridge-safe PyTorch 2.13.0 build using
OpenBLAS without MKL/MKLDNN; subsequent stress and generation campaigns had
zero new SIGILL events. This describes the tested environment and is not a
general MKL claim.

The largest current performance limitation is `QuantLinear`: the checkpoint
weights are FP32 and the reference path repeats fake quantization each forward,
rather than using true INT8 storage and an INT8 CPU GEMM backend. Generation is
therefore slow (roughly 14 s/token in the short-prompt tests).

Repository and detailed engineering reports:

https://github.com/Uraroga/spikingbrain-cpu-cluster

- Would a CPU/reference fallback be useful to the project?
- Are the maintainers aware of related CPU W8ASpike work?
- What would you recommend for true INT8 CPU storage/GEMM, and what parts might
  be appropriate for an upstream contribution?

This is an independent community experiment; no priority claim is intended.
