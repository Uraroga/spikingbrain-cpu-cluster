# Goal 12B — Physical Ethernet cable loss during active Gloo generation

## Outcome

**TIMEOUT FAILURE (dual-rank, clean process exit).**

Physically disconnecting `eno1` while both hosts remained powered on did not
produce a TCP closure immediately observable by Gloo. Rank 1 remained blocked
in `recv` and rank 0 in `send`; both reached the configured 75-second timeout
and terminated autonomously with exit code 1. No processes, listeners, or
containers were left running.

## Experiment constraints and integrity

- Only fault: Sergio manually disconnected the private Ethernet cable.
- No firewall, `ip link down`, `kill`, `init 0`, shutdown, or software fault
  injection.
- No code or repository changes; no commit or push; worktree remained clean.
- Exclusive runtime: `spikingbrain-cpu:goal8.6-openblas`.
- Gloo bound to `eno1`; remote management of argo3 maintained over Wi-Fi.
- Ranks and monitors were not interrupted manually.
- No generation was restarted after reconnecting the cable.

## Preflight

### Network

Before the test, both nodes reported:

- `/sys/class/net/eno1/carrier = 1`;
- `/sys/class/net/eno1/operstate = up`;
- flags `UP,LOWER_UP`, MTU 1500;
- direct private route over `eno1`.

| Function | Atlas5 | Argo3 |
|---|---|---|
| Gloo Ethernet | `eno1`, 192.168.1.128 | `eno1`, 192.168.1.64 |
| Wi-Fi management | 192.168.0.89 | 192.168.0.247 |

The SSH check explicitly records:

```text
SSH_CONNECTION=192.168.0.89 ... 192.168.0.247 22
```

On the same remote host, `ip route get 192.168.1.128` instead reported
`dev eno1 src 192.168.1.64`. Therefore management traffic and Gloo traffic used
different paths. After link-down, the Wi-Fi session continued to work.

### Runtime, sources, and resources

- Image on both hosts:
  `sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836`.
- Stable Ivy Bridge OpenBLAS PyTorch runtime, no MKL/MKLDNN/CUDA.
- `distributed_generate.py` hash on both nodes:
  `b37eb401ab870ae2de6747934c099598d78abf27e9e9aa3fd295140377dcf9a3`.
- `protocol.py` hash on both nodes:
  `2cfdb5730b02a617efd9eecbc3254e5252f7de3dacc064f5a3cfd93fa92b8962`.
- Checkpoint and tokenizer present and mounted read-only.
- About 28–30 GiB of memory available, swap 0, and ample disk space.
- Port 29622 free; no Goal 12 process or container running.
  Only the stopped Goal 12A containers were preserved.

## Exact configuration

| Item | Value |
|---|---|
| Date | 2026-08-16, Europe/Rome (CEST, UTC+02:00) |
| Backend | CPU Gloo, world size 2 |
| Rendezvous | 192.168.1.128:29622 |
| Gloo interface | `eno1` |
| Docker networking | host |
| Partition | atlas5: embedding/layers 0–13; argo3: layers 14–27/norm/`lm_head` |
| Threads | 4 per rank |
| Timeout | 75 s |
| `max_prompt_tokens` | 64 |
| `max_new_tokens` | 40 |
| `max_bytes` | 16,000,000,000 per rank |
| Tokenized prompt | 31 |
| Decoding | greedy argmax |

Prompt:

> Explain in detail how two old computers can cooperate to run a large language model, describing the flow of data between the machines and the main technical limitations.

## Rank commands

Rank 1 on argo3, started through the Wi-Fi management path but with Gloo on `eno1`:

```bash
docker run --name goal12b-argo-rank1 --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike:/model:ro \
  -v /home/sergio/goal12a-20260816/runtime/src:/app/src:ro \
  -v /home/sergio/goal12a-20260816/runtime/scripts:/app/scripts:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_argo.py \
  --master-addr 192.168.1.128 --port 29622 --peer 192.168.1.128 \
  --model-dir /model --max-bytes 16000000000 \
  --max-prompt-tokens 64 --max-new-tokens 40 --threads 4 --timeout 75
```

Rank 0 on atlas5:

```bash
docker run --name goal12b-atlas-rank0 --network host \
  -e GLOO_SOCKET_IFNAME=eno1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-atlas:/model:ro \
  -v /home/sergio/Modelli/SpikingBrain/SpikingBrain-7B-W8ASpike-tokenizer:/tokenizer:ro \
  -v /home/sergio/Progetti/spikingbrain-cluster/src:/app/src:ro \
  -v /home/sergio/Progetti/spikingbrain-cluster/scripts:/app/scripts:ro \
  spikingbrain-cpu:goal8.6-openblas \
  python scripts/generate_atlas.py \
  --master-addr 192.168.1.128 --port 29622 --peer 192.168.1.64 \
  --model-dir /model --tokenizer-dir /tokenizer \
  --prompt 'Explain in detail how two old computers can cooperate to run a large language model, describing the flow of data between the machines and the main technical limitations.' \
  --max-bytes 16000000000 --max-prompt-tokens 64 \
  --max-new-tokens 40 --threads 4 --timeout 75
```

## Monitors and timing quality

Before starting the ranks, the following were started:

- high-frequency carrier/operstate/link monitors on both hosts;
- Ethernet ping, container, PID, CPU, and RSS monitors;
- joint observer queried through Wi-Fi;
- kernel journal/NetworkManager as an independent source.

The 10 Hz carrier monitor on argo3 is complete. The dedicated atlas5 monitor
terminated prematurely before workload execution because of a local orchestration
session limitation; this anomaly is preserved in the logs and is not hidden.
Atlas5 remains covered by the joint observer and, most importantly, by the kernel
journal at microsecond precision. The fault time is never derived from the time
when Sergio was instructed to disconnect the cable.

10 Hz bracket on argo3:

```text
15:16:26.319336804  carrier=1 operstate=up
15:16:26.452333899  carrier=0 operstate=down
```

Bracket precision: 132.997 ms. Operational timestamp used for latency
measurements: **15:16:26.452333899 CEST**, first certain sample at carrier 0.

Kernel confirmations:

- argo3: `NIC Link is Down` at **15:16:26.333509**;
- atlas5: `NIC Link is Down` at **15:16:26.730568**.

The small difference reflects NIC detection/clock behavior and does not affect
the classification.

## Proof of generation before the fault

Sergio was instructed to disconnect the cable only after the following sequence
had been observed:

1. both process groups initialized;
2. atlas stage materialized (~13.5 GiB RSS during forward);
3. argo stage materialized (~15.6 GiB RSS during forward);
4. Atlas prefill sent and real Argo forward completed;
5. token 1 received by Atlas;
6. subsequent Atlas decode completed and new hidden state delivered to Argo.

At carrier-down, **2 tokens** had completed. Rank 0 was locally processing token
2 to produce the next hidden state. Token IDs/text cannot be recovered: the
driver inserts the steps only into the final success JSON, which is not produced
on the exception path. The count is derived from the alternating CPU/Gloo
sequence and the traceback positions.

## Fault timeline

| Timestamp CEST | Event |
|---|---|
| 15:14:33.358 | Rank 1 container started |
| 15:14:35.213 | Rank 0 container started |
| 15:16:05–15:16:13 | Token 1 received/processed; next autoregressive decode started on Argo |
| ~15:16:22.987 | Rank 1 entered `recv` for the next hidden header |
| 15:16:26.319 | Last carrier 1 sample on argo3 |
| 15:16:26.452 | First carrier 0/down sample on argo3: adopted fault time |
| 15:16:26.731 | atlas5 kernel: NIC link down |
| ~15:16:29.682 | Rank 0 entered `send` for the hidden payload |
| 15:17:37.987 | Rank 1 emitted the first Gloo error |
| 15:17:38.600 | Rank 1 container terminated, exit 1 |
| 15:17:44.682 | Rank 0 emitted its Gloo error |
| 15:17:45.438 | Rank 0 container terminated, exit 1 |
| 15:19:48.335 | argo3 link up again, 1 Gbps full duplex |
| 15:19:48.977 | atlas5 link up again, 1 Gbps full duplex |

Estimated operation-entry times are derived from the error timestamp minus the
75.000 s explicitly reported by Gloo.

## First detecting rank and Gloo operations

### Rank 1 / argo3 — first failure

Operation:

```text
argo() -> recv_hidden() -> dist.recv(header, peer) -> work.wait()
```

Exact error:

```text
RuntimeError: [/src/third_party/gloo/gloo/transport/tcp/unbound_buffer.cc:78]
Timed out waiting 75000ms for recv operation to complete
```

- Error emitted: 15:17:37.986695477.
- Latency from first carrier 0: **71.534361578 s**.
- The operation had already been waiting for about 3.466 s when the cable was
  disconnected; this is why latency from carrier-down is less than 75 s.
- Exit code: **1**, `OOMKilled=false`.

### Rank 0 / atlas5 — second failure

`send_hidden()` sends a header first and then the payload. The traceback is on
the second send:

```text
atlas() -> send_hidden(next_hidden, 1)
        -> dist.send(hidden.contiguous(), peer) -> work.wait()
```

Exact error:

```text
RuntimeError: [/src/third_party/gloo/gloo/transport/tcp/unbound_buffer.cc:129]
Timed out waiting 75000ms for send operation to complete
```

- Error emitted: 15:17:44.681905653.
- Latency from first carrier 0: **78.229571754 s**.
- The blocking `send` started about 3.230 s after carrier-down, after rank 0 had
  finished its local computation.
- Exit code: **1**, `OOMKilled=false`.

Rank 1 therefore reported the problem **6.695210176 s** before rank 0, not
because of a different detection mechanism, but because its `recv` had started
before rank 0's `send`. Both waited exactly for the timeout of their respective
operation.

## Timeout and TCP/Gloo behavior

Loss of carrier did not generate `Connection closed by peer`: both hosts and
processes were still alive and no endpoint had orderly closed the socket.
Operations remained waiting until:

- `Timed out waiting 75000ms for recv operation to complete` on rank 1;
- `Timed out waiting 75000ms for send operation to complete` on rank 0.

The 75 s timeout was therefore reached on both ranks. This is not an immediate
link failure and not an indefinite hang.

## `shutdown()` / barrier / destroy process group

On both ranks the driver:

1. catches the application exception;
2. prints the error JSON;
3. enters `finally` and calls `shutdown()`;
4. attempts `barrier()` and guarantees `destroy_process_group()`;
5. re-raises the original exception.

Observed timings:

- rank 1: JSON at 15:17:37.9867, re-raised traceback at 15:17:37.9929
  (~6 ms), container terminated at 15:17:38.5999;
- rank 0: JSON at 15:17:44.6819, traceback at 15:17:44.6836 (~2 ms),
  container terminated at 15:17:45.4381.

The final barrier did not add a second timeout or hang; the already-failed
process group caused the barrier to fail/return quickly and
`destroy_process_group()` was executed. As in Goal 12A, the exact
`shutdown_error` text is not visible because the report is printed before the
`finally` block adds that field and is not printed again on the re-raise path.

## Cleanup and native anomalies

Before reconnection and after restoration:

- no `distributed_generate`, `generate_atlas`, or `generate_argo` process;
- no listener on port 29622;
- no running Goal 12B container;
- stopped containers preserved for inspection: both `Exited (1)`;
- swap 0 on both;
- `SIGILL`: 0;
- `invalid opcode`: 0;
- segfault/trap/`libtorch_cpu.so`: 0;
- OOM/`Killed process`: 0.

After restoration, both hosts reported `carrier=1`, `operstate=up`, direct
192.168.1.x route on `eno1`, and Ethernet ping 3/3 (~0.3 ms). The argo3 control
session continued to report `SSH_CONNECTION` over the Wi-Fi network.

## Goal 12A / Goal 12B comparison

| Aspect | Goal 12A: host shutdown | Goal 12B: physical cable |
|---|---|---|
| Peer host | Orderly shutdown, process eventually terminated | Both hosts remained powered on |
| TCP state | Peer/process closes or disappears | Socket alive but path has no carrier |
| First error | Rank 0, `Connection closed by peer` | Rank 1, `recv` timeout |
| Other rank | Rank 1 terminated by shutdown (137) | Rank 0 `send` timeout |
| Gloo timeout | Not reached | Reached on both, 75.000 s |
| Relevant latency | ~2.612 s from actual rank 1 death to rank 0 error | 71.534 s carrier→rank 1; 78.230 s carrier→rank 0 |
| Classification | CLEAN FAILURE | TIMEOUT FAILURE |
| Residuals/native crashes | None | None |

The main difference is causal: Goal 12A produces a peer closure visible to the
TCP stack; Goal 12B produces a silent loss of the physical medium, so Gloo can
only wait for the operation timeout to expire.

## Artifacts

Directory: `/home/sergio/goal12b-20260816/`

- atlas5 and argo3 preflight via Wi-Fi;
- proof of separate routes and source hashes;
- clock comparison;
- `valid/config.txt`;
- separate stdout, stderr, timestamps, and exit codes for each rank;
- timestamped Docker logs and full inspect output;
- `carrier_argo.tsv` at 10 Hz;
- joint observer and state/ping/CPU/RSS monitors;
- kernel and NetworkManager link-down/link-up logs;
- cleanup checks before/after reconnection.

## Conclusion

**TIMEOUT FAILURE.** Physical Ethernet loss during active generation is not
detected immediately by Gloo in this configuration. Rank 1, blocked on `recv`
for the next header, and rank 0, blocked on `send` of the hidden payload, each
expire after 75 seconds from the start of their own operation. Rank 1 terminates
first because its wait began before link-down. Both processes exit cleanly
(code 1), the final barrier does not introduce a second block, no active
resources remain, and no native crashes emerge.
