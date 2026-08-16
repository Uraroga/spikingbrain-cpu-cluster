# Goal 12A — Gloo controlled peer-loss test

## Outcome

**CLEAN FAILURE (peer loss detected without a hang and without waiting for the Gloo timeout).**

Atlas5/rank 0 detected the loss of argo3/rank 1 at the first new Gloo operation
performed after completing its own local computation. The error was
`Connection closed by peer` during the `dist.send()` of the header for the
next autoregressive hidden state. Rank 0 terminated autonomously with exit code 1;
it was not interrupted manually. No processes, listeners, or containers were left
running.

## Constraints respected

- No changes to the code or repository; final worktree clean.
- No commit, push, or update to the public repository during the experiment.
- Shutdown was performed manually by Sergio with `sudo init 0` only after
  confirming that autoregressive generation had started and at least one token had
  completed successfully.
- No `kill`, fault injection, or automatic restart was used to trigger the failure.
- Exclusive runtime: `spikingbrain-cpu:goal8.6-openblas`.
- Checkpoint, tokenizer, `src/`, and `scripts/` mounted read-only.
- Private Ethernet `eno1`, MTU 1500, Docker host networking, and CPU Gloo.

## Observed implementation and `--timeout` semantics

Inspected commit: `eba802a` (`eba802a...`, full hash recorded in preflight).

`scripts/distributed_generate.py` exposes `--timeout` in seconds (default 180)
and passes it to `initialize()`. `protocol.initialize()` converts it to
`timedelta(seconds=timeout_seconds)` and assigns it to
`dist.init_process_group("gloo", ...)`. The final `send`, `recv`, and `barrier`
calls do not set individual timeouts: they use the process-group timeout.

`shutdown()` first executes `dist.barrier()` and then guarantees
`dist.destroy_process_group()` in a `finally`.

## Configuration of the valid test (attempt 3)

| Item | Value |
|---|---|
| Date | 2026-08-16, Europe/Rome (CEST, UTC+02:00) |
| Rank 0 | atlas5, 192.168.1.128, Intel i5-4590 |
| Rank 1 | argo3, 192.168.1.64, Intel i3-3240 |
| Interface | `eno1`, Ethernet, MTU 1500 |
| Backend | Gloo, world size 2, port 29612 |
| Docker | `--network host` |
| Image | `spikingbrain-cpu:goal8.6-openblas` |
| Image ID | `sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836` on both hosts |
| PyTorch | `2.13.0+openblas.ivybridge`, MKL/MKLDNN/CUDA disabled |
| Partition | atlas5 embedding + layers 0–13; argo3 layers 14–27 + norm + `lm_head` |
| Threads | 4 per rank |
| Timeout | 75 s |
| `max_prompt_tokens` | 64 |
| `max_new_tokens` | 20 |
| `max_bytes` | 16,000,000,000 per rank |
| Decoding | greedy argmax |
| Tokenized prompt | 31 tokens |

Exact prompt:

> Explain in detail how two old computers can cooperate to run a large language model, describing the flow of data between the machines and the main technical limitations.

### Why 75 seconds instead of the preferred 30

The 30 s attempt exposed a startup limitation: rank 1 completes its stage mmap
in about 0.6 s and immediately waits for the first hidden state, while rank 0
still has to materialize its own stage and complete prefill. Rank 1's `recv`
timed out after 30.000 s before rank 0 could send the prefill. This was not a
peer-loss event and is not counted as the valid test. A 75 s timeout was chosen,
the minimum prudent value relative to the observed path while still being far
below the historical 180 s setting. During the real fault, the 75 s timeout was
not reached.

## Exact commands for the valid test

Rank 1, started first via SSH on argo3:

```bash
docker run --name goal12a-argo-rank1 --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike:/model:ro \
  -v /home/sergio/goal12a-20260816/runtime/src:/app/src:ro \
  -v /home/sergio/goal12a-20260816/runtime/scripts:/app/scripts:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_argo.py \
  --master-addr 192.168.1.128 --port 29612 --peer 192.168.1.128 \
  --model-dir /model --max-bytes 16000000000 \
  --max-prompt-tokens 64 --max-new-tokens 20 --threads 4 --timeout 75
```

Rank 0, started on atlas5:

```bash
docker run --name goal12a-atlas-rank0 --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-atlas:/model:ro \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-tokenizer:/tokenizer:ro \
  -v /home/sergio/Progetti/spikingbrain-cluster/src:/app/src:ro \
  -v /home/sergio/Progetti/spikingbrain-cluster/scripts:/app/scripts:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_atlas.py \
  --master-addr 192.168.1.128 --port 29612 --peer 192.168.1.64 \
  --model-dir /model --tokenizer-dir /tokenizer \
  --prompt 'Explain in detail how two old computers can cooperate to run a large language model, describing the flow of data between the machines and the main technical limitations.' \
  --max-bytes 16000000000 --max-prompt-tokens 64 \
  --max-new-tokens 20 --threads 4 --timeout 75
```

## Timeline

| Timestamp CEST | Observed event |
|---|---|
| 14:57:20.372 | Rank 1 container started |
| 14:57:21.909 | Rank 0 container started |
| around 14:57:38 | Rank 0 stage materialized; Atlas prefill completed and rank 1 started the real forward |
| 14:58:48–14:58:53 | Token 1 received by rank 0 and first Atlas autoregressive decode completed |
| 14:58:54–14:59:03 | Rank 1 processed the next step; token 2 completed |
| 14:59:04–14:59:09 | Rank 0 processed token 2 and sent the next hidden state |
| 14:59:10 | Rank 1 started computing token 3 |
| 14:59:11 | Sergio executed `sudo init 0`; systemd recorded `The system will power off now!` |
| 14:59:11.257 | Docker received SIGTERM as part of host shutdown |
| 14:59:11.686 | First SSH sample: argo3 unreachable |
| around 14:59:18 | Rank 0 started processing token 3, which had already been delivered during the shutdown window |
| 14:59:21.333 | Docker: container did not exit within 10 s after SIGTERM; forced termination applied |
| 14:59:21.552 | Rank 1 terminated with exit 137, `OOMKilled=false` |
| 14:59:24.164 | Rank 0 emitted JSON error `Connection closed by peer` |
| 14:59:24.944 | Rank 0 container terminated, exit 1 |
| 14:59:22 | End of previous boot / system power off |
| 15:00:41 | Current argo3 boot started |

## Fault point and completed tokens

At the moment shutdown started, **two tokens had definitely completed**
and rank 1 was computing the third. Because Docker waited 10 seconds after
SIGTERM before forced termination, the third token was produced and delivered
during the shutdown sequence; rank 0 received and processed it.

Therefore:

- tokens fully completed before the fault command: **2**;
- token 3: in progress when the fault started, but produced/received before the
  actual termination of the rank 1 process;
- failure occurs while sending the header of the hidden state derived from token 3.

The token IDs/text cannot be recovered from rank 1: the implementation serializes
them only in the final success JSON, which was not emitted because host shutdown
terminated the process. The count is supported by the alternating timestamped CPU
counters and by the rank 0 traceback location.

## Atlas5 behavior and peer-loss latency

Exact exception:

```text
RuntimeError: [/src/third_party/gloo/gloo/transport/tcp/pair.cc:553]
Connection closed by peer [192.168.1.64]:62885.
```

Application stack:

```text
atlas() -> send_hidden(next_hidden, 1) -> dist.send(header, peer)
       -> isend(...) -> ProcessGroupGloo.send(...)
```

Latency measurements:

- from Docker shutdown start (`14:59:11.257`) to rank 0 exception
  (`14:59:24.164`): **12.907 s**;
- from first sample where argo3 was unreachable (`14:59:11.686`) to exception:
  **12.477 s**;
- from actual termination of the rank 1 process (`14:59:21.552`) to exception:
  **2.612 s**.

Most of the first duration was not spent blocked inside Gloo: rank 0 was still
performing local computation on token 3, which had already been received. The
next `send` detected the closed connection quickly and did not wait 75 s.

## Timeout

- 30 s: insufficient for the asymmetric startup path; normal timeout on rank 1
  `recv` before prefill. This attempt is not valid as a peer-loss test.
- 75 s in the valid test: **not reached**.
- Peer loss: reported through explicit TCP closure (`Connection closed by peer`),
  not `Timed out waiting ...`.

## `shutdown()` and barrier

After the application exception, the driver prints the error JSON and enters the
`finally` block, where it calls `shutdown()`, i.e. `barrier()` followed by
`destroy_process_group()`.

The error JSON is timestamped `14:59:24.1637`; the re-raised traceback starts at
`14:59:24.1680`, about 4 ms later. The container terminates at `14:59:24.9436`.

Therefore, the final barrier **did not remain blocked and did not wait for the
timeout**; with the peer absent it returned/failed immediately and the process
group was destroyed. The exact `shutdown_error` text is not observable: the code
adds that field to the report after the error JSON has already been printed and,
on the re-raise path, does not print the report again. This is a limitation of
the current telemetry, not a requested or implemented code change.

## Cleanup

### Atlas5

- rank 0: exit code **1**, `OOMKilled=false`;
- no residual `distributed_generate`/`generate_atlas` process;
- no listener on port 29612;
- no running Goal 12A container;
- stopped container intentionally preserved for inspection: `Exited (1)`;
- host swap 0.

### Argo3 after reboot

- rank 1: Docker exit code **137**, `OOMKilled=false`;
- restart policy `no`, therefore no automatic workload restart;
- no residual generation process;
- no listener on port 29612;
- no running Goal 12A container;
- stopped container intentionally preserved for inspection: `Exited (137)`;
- host swap 0.

Exit 137 does not indicate OOM in this case: the journal proves that Docker,
during orderly host shutdown, sent SIGTERM, waited 10 s, and then forcibly
terminated the container before completing poweroff.

## Kernel and cause separation

In the previous argo3 boot:

- `SIGILL`: 0;
- `invalid opcode`: 0;
- `segfault`: 0;
- traps or references to `libtorch_cpu.so`: 0;
- OOM kill / `Killed process`: 0.

The only line containing “OOM” is the normal startup of the `systemd-oomd`
socket, not a memory event. The journal explicitly records the user-requested
shutdown and the systemd/Docker sequence; there is no evidence of an unrelated
native crash.

## Artifacts

Main directory: `/home/sergio/goal12a-20260816/`

- `preflight_atlas.txt`, `preflight_argo.txt`, prompt tokenization;
- `attempt2/`: startup-limit test with 30 s timeout;
- `attempt3/rank0.stdout`, `rank0.stderr`, separate exit/timestamp files;
- `attempt3/rank1.stdout`, `rank1.stderr`, post-boot Docker state;
- `attempt3/live_monitor.tsv`: timestamped RSS/CPU for both ranks;
- timestamped Docker logs;
- complete journal and kernel logs from the previous argo3 boot;
- cleanup checks on both nodes;
- hashes of the source files mounted on both hosts.

The first attempt, also preserved, failed before model execution because the
container root filesystem had been set read-only and PyTorch could not find a
writable temporary directory. It produced neither generation nor fault and does
not contribute to the Goal 12A outcome.

## Conclusion

**CLEAN FAILURE.** Controlled shutdown of rank 1 does not leave rank 0 hanging.
Rank 0 can finish local computation already fed by previously received data,
then the next Gloo `send` detects the closed peer in about 2.6 s from the actual
termination of rank 1 (about 12.9 s from the beginning of host shutdown),
without reaching the configured timeout. The final barrier/destroy path also
adds no observable blocking. No SIGILL or native anomaly is associated with the
result.
