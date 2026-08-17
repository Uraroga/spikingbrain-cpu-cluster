#!/usr/bin/env python3
"""Persistent two-rank terminal chat using the validated CPU model partition."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.distributed as dist

from distributed_generate import cache_report, ensure_caches, recv_hidden, send_hidden
from distributed_stage import check_memory, load_stage, memory
from spikingbrain_cpu.protocol import initialize, shutdown

COMMAND_GENERATE = 1
COMMAND_SHUTDOWN = 2
VOCAB_SIZE = 152064


def send_command(command: int, prompt_length: int = 0, max_new_tokens: int = 0) -> None:
    dist.send(torch.tensor([command, prompt_length, max_new_tokens], dtype=torch.int64), 1)


def receive_command() -> tuple[int, int, int]:
    header = torch.empty(3, dtype=torch.int64)
    dist.recv(header, 0)
    command, prompt_length, max_new_tokens = map(int, header.tolist())
    if command not in (COMMAND_GENERATE, COMMAND_SHUTDOWN):
        raise ValueError(f"invalid chat command {command}")
    if command == COMMAND_GENERATE and (prompt_length < 1 or max_new_tokens < 1):
        raise ValueError(f"invalid generation command values: {header.tolist()}")
    return command, prompt_length, max_new_tokens


def render_prompt(tokenizer, messages: list[dict[str, str]], limit: int) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if not 0 < len(ids) <= limit:
        raise ValueError(
            f"conversation is {len(ids)} prompt tokens; configured limit is {limit}. "
            "Use /reset or restart with a larger --max-prompt-tokens. Nothing was truncated."
        )
    if any(not 0 <= token < VOCAB_SIZE for token in ids):
        raise ValueError("prompt token outside model vocabulary")
    return ids


def atlas_turn(
    args, stage, tokenizer, prompt_ids: list[int], emit_text: bool = True,
    stop_requested=lambda: False,
):
    caches: dict = {}
    generated: list[int] = []
    steps: list[dict] = []
    started = time.perf_counter()
    send_command(COMMAND_GENERATE, len(prompt_ids), args.max_new_tokens)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        hidden = stage.embed(ids)
        hidden, caches, _, _ = stage.forward_layers(hidden, caches)
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("atlas hidden contains NaN/Inf after prefill")
    ensure_caches(caches, range(14), len(prompt_ids))
    prefill_ms = (time.perf_counter() - prefill_started) * 1000
    send_hidden(hidden, 1)
    displayed = ""
    stop_reason = "max_new_tokens"
    while len(generated) < args.max_new_tokens:
        token_tensor = torch.empty(1, dtype=torch.int64)
        wait_started = time.perf_counter()
        dist.recv(token_tensor, 1)
        token = int(token_tensor.item())
        if not 0 <= token < len(tokenizer):
            raise ValueError(f"generated token outside tokenizer range: {token}")
        generated.append(token)
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if emit_text:
            suffix = decoded[len(displayed):] if decoded.startswith(displayed) else decoded
            print(suffix, end="", flush=True)
        displayed = decoded
        stop = (
            token == tokenizer.eos_token_id
            or len(generated) >= args.max_new_tokens
            or stop_requested()
        )
        if token == tokenizer.eos_token_id:
            stop_reason = "eos"
        dist.send(torch.tensor([0 if stop else 1], dtype=torch.int64), 1)
        step = {
            "token_id": token,
            "token_wait_ms": (time.perf_counter() - wait_started) * 1000,
        }
        if not stop:
            decode_started = time.perf_counter()
            with torch.inference_mode():
                hidden = stage.embed(torch.tensor([[token]], dtype=torch.long))
                hidden, caches, _, _ = stage.forward_layers(hidden, caches)
            if not torch.isfinite(hidden).all():
                raise FloatingPointError("atlas hidden contains NaN/Inf during decode")
            ensure_caches(caches, range(14), len(prompt_ids) + len(generated))
            step["atlas_decode_ms"] = (time.perf_counter() - decode_started) * 1000
            send_hidden(hidden, 1)
        steps.append(step)
        if stop:
            break
    return displayed, {
        "prompt_tokens": len(prompt_ids),
        "generated_ids": generated,
        "generated_text": displayed,
        "stop_reason": stop_reason,
        "prefill_ms": prefill_ms,
        "total_ms": (time.perf_counter() - started) * 1000,
        "cache": cache_report(caches),
        "steps": steps,
        "memory": memory(),
    }


def atlas(args, stage):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import Qwen2Tokenizer

    tokenizer_started = time.perf_counter()
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True
    )
    if not tokenizer.chat_template:
        raise RuntimeError("tokenizer has no chat template")
    tokenizer_load_ms = (time.perf_counter() - tokenizer_started) * 1000
    messages: list[dict[str, str]] = []
    turns = []
    signal_state = {"in_turn": False, "stop_requested": False}
    previous_sigint = signal.getsignal(signal.SIGINT)

    def on_sigint(_signum, _frame):
        signal_state["stop_requested"] = True
        if not signal_state["in_turn"]:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)
    print("SpikingBrain persistent chat")
    print("Type /reset to clear conversation")
    print("Type /quit to exit\n")
    try:
        while True:
            try:
                user_text = input("Tu> ")
            except EOFError:
                break
            command = user_text.strip()
            if command == "/quit":
                break
            if command == "/reset":
                messages.clear()
                print("Conversation cleared.\n")
                continue
            if not command:
                continue
            candidate = messages + [{"role": "user", "content": user_text}]
            try:
                prompt_ids = render_prompt(tokenizer, candidate, args.max_prompt_tokens)
            except ValueError as exc:
                print(f"SpikingBrain> Error: {exc}\n")
                continue
            print("SpikingBrain> ", end="", flush=True)
            signal_state["in_turn"] = True
            assistant_text, turn = atlas_turn(
                args,
                stage,
                tokenizer,
                prompt_ids,
                stop_requested=lambda: signal_state["stop_requested"],
            )
            signal_state["in_turn"] = False
            print("\n")
            messages = candidate + [{"role": "assistant", "content": assistant_text}]
            turns.append(turn)
            if signal_state["stop_requested"]:
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        send_command(COMMAND_SHUTDOWN)
    return {
        "turn_count": len(turns),
        "turns": turns,
        "tokenizer_load_ms": tokenizer_load_ms,
        "stage_load_count": 1,
        "memory": memory(),
    }


def argo_turn(args, stage, prompt_length: int, max_new_tokens: int):
    caches: dict = {}
    generated = []
    steps = []
    started = time.perf_counter()
    while len(generated) < max_new_tokens:
        hidden, recv_ms = recv_hidden(0, args.max_prompt_tokens)
        expected_length = prompt_length if not generated else 1
        if hidden.shape[1] != expected_length:
            raise ValueError(
                f"hidden length {hidden.shape[1]} does not match expected {expected_length}"
            )
        expected_position = prompt_length + len(generated)
        layer_started = time.perf_counter()
        with torch.inference_mode():
            hidden, caches, _, _ = stage.forward_layers(hidden, caches)
        ensure_caches(caches, range(14, 28), expected_position)
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("argo hidden contains NaN/Inf")
        normalized = stage.apply_final_norm(hidden[:, -1:])
        logits = stage.project_logits(normalized)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("argo logits contain NaN/Inf")
        token = torch.argmax(logits[:, -1, :], dim=-1).to(torch.int64)
        token_id = int(token.item())
        if not 0 <= token_id < VOCAB_SIZE:
            raise ValueError(f"generated token outside model vocabulary: {token_id}")
        dist.send(token.reshape(1), 0)
        generated.append(token_id)
        control = torch.empty(1, dtype=torch.int64)
        dist.recv(control, 0)
        sample = memory()
        check_memory(sample, args.max_rss_mib)
        steps.append({
            "token_id": token_id,
            "recv_ms": recv_ms,
            "layers_and_head_ms": (time.perf_counter() - layer_started) * 1000,
            "memory": sample,
        })
        if not int(control.item()):
            break
    return {
        "prompt_tokens": prompt_length,
        "generated_ids": generated,
        "total_ms": (time.perf_counter() - started) * 1000,
        "cache": cache_report(caches),
        "steps": steps,
        "memory": memory(),
    }


def argo(args, stage):
    turns = []
    while True:
        command, prompt_length, max_new_tokens = receive_command()
        if command == COMMAND_SHUTDOWN:
            break
        if prompt_length > args.max_prompt_tokens:
            raise ValueError(
                f"prompt has {prompt_length} tokens; limit is {args.max_prompt_tokens}"
            )
        turns.append(argo_turn(args, stage, prompt_length, max_new_tokens))
    return {
        "turn_count": len(turns),
        "turns": turns,
        "stage_load_count": 1,
        "memory": memory(),
    }


def main(default_rank=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=default_rank)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--peer", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-bytes", type=int, default=16_000_000_000)
    parser.add_argument("--max-rss-mib", type=float, default=23552)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.rank not in (0, 1):
        parser.error("--rank is required")
    if args.rank == 0 and args.tokenizer_dir is None:
        parser.error("--tokenizer-dir is required on rank 0")
    if args.max_prompt_tokens < 1 or args.max_new_tokens < 1:
        parser.error("token limits must be positive")
    torch.set_num_threads(args.threads)
    initialized = False
    report = {"rank": args.rank, "success": False}
    try:
        report["network"] = initialize(
            args.rank, args.master_addr, args.port, args.peer, args.timeout
        )
        initialized = True
        stage, report["loading"] = load_stage(args.rank, args)
        report["session"] = atlas(args, stage) if args.rank == 0 else argo(args, stage)
        report["success"] = True
    except KeyboardInterrupt:
        report["interrupted"] = True
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        if initialized:
            try:
                shutdown()
                report["clean_shutdown"] = True
            except Exception as exc:
                report["clean_shutdown"] = False
                report["shutdown_error"] = str(exc)
        print("CHAT_REPORT " + json.dumps(report), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
