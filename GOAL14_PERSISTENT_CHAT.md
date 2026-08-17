# Goal 14: Persistent two-node terminal chat

## Objective

Goal 14 turns the verified two-node generation path into a persistent terminal
chat session. It preserves the existing 14/14 model partition and stable
OpenBLAS runtime; it does not introduce a new inference architecture or Docker
image.

## Architecture

Atlas5 remains rank 0 and owns tokenization, embeddings, layers 0–13, the
terminal UI, and generated-text decoding. Argo3 remains rank 1 and owns layers
14–27, final normalization, the language-model head, and greedy token
selection. The ranks exchange FP32 hidden states and `int64` token IDs through
the persistent Gloo process group.

## Persistent model loading

Each process initializes its distributed rank and loads its model stage exactly
once. Rank 0 also loads the local tokenizer once. The terminal loop handles
multiple turns without recreating the processes, process group, model stages,
or tokenizer. The measured session reported `stage_load_count = 1` on both
ranks.

## Protocol extension

Rank 0 sends a fixed three-element command header before work begins:

```text
GENERATE, prompt_length, max_new_tokens
SHUTDOWN, 0, 0
```

`GENERATE` starts one complete turn and gives rank 1 the expected prefill
length and decode bound. `SHUTDOWN` ends rank 1's receive loop so that both
ranks can tear down the process group cleanly.

## Per-turn conversation and caches

For every user message, rank 0 reconstructs the full conversation and renders
it with the tokenizer's real chat template and generation prompt. The rendered
prompt is rejected explicitly if it exceeds the configured context limit; it
is never silently truncated.

Both ranks create fresh KV/GLA cache dictionaries for each conversation turn.
The complete rendered conversation is therefore prefetched again and its cache
state rebuilt for that turn. Decode steps reuse those per-turn caches, while
the already loaded model weights remain resident across all turns.

## Terminal UI

The terminal runs on atlas5. `/reset` clears the in-memory conversation history
without reloading the model stages, and `/quit` requests a clean distributed
shutdown. EOF and Ctrl-C also follow the shutdown path. Empty input is ignored.
Maximum prompt tokens and maximum generated tokens are configurable.

## Deployment to argo3

`scripts/run_chat_cluster.sh` verifies the exact stable image ID on both hosts,
copies `src/` and only the required Python entry points to a configurable
remote staging directory, starts rank 1 on argo3, and runs the interactive rank
0 container on atlas5. Project code and model/tokenizer directories are mounted
read-only. Host names, addresses, network interface, paths, port, and token
limits are configurable through environment variables; see
[CHAT_README.md](CHAT_README.md).

The validated runtime remains unchanged:

```text
spikingbrain-cpu:goal8.6-openblas
sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836
```

## Verification

The local suite completed with 44 passing tests. A real two-node smoke session
completed two user turns and generated eight tokens per turn. Prompt lengths
were 25 tokens for turn 1 and 52 tokens for turn 2. Generated token IDs matched
across ranks, and all finite hidden-state and logit checks passed.

Cache positions were coherent across all 28 layers:

| turn | prompt tokens | generated tokens | final cache position |
|---:|---:|---:|---:|
| 1 | 25 | 8 | 32 |
| 2 | 52 | 8 | 59 |

Swap remained 0 MiB on both hosts. No new SIGILL, invalid-opcode, or
`libtorch_cpu` trap was observed.

## Measured smoke values

These values describe this specific two-turn smoke session and are not a
general performance benchmark.

| host | turn 1 | turn 2 | final RSS | peak RSS (HWM) | swap |
|---|---:|---:|---:|---:|---:|
| atlas5 | 121.39 s | 128.97 s | 13,243.51 MiB | 13,245.00 MiB | 0 MiB |
| argo3 | 121.39 s | 128.97 s | 15,141.54 MiB | 15,218.97 MiB | 0 MiB |

## Clean shutdown

Both rank reports recorded `clean_shutdown = true`. After the smoke session,
no chat containers remained and no listener remained on port 29615.

## Limitations

Persistent chat removes model reloads between turns, but inference remains very
slow because CPU fake-quantized `QuantLinear` computation is still the dominant
bottleneck. Each new user turn also re-renders and prefills the complete
conversation with fresh caches, so turn latency grows with conversation length.
The implementation uses deterministic greedy generation, has only been tested
on the documented two-host setup, and is experimental rather than a hardened
serving system.
