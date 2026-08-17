# Persistent two-node terminal chat

`scripts/run_chat_cluster.sh` verifies the immutable stable image ID on both
hosts, copies only `src/` and the four required Python scripts to argo3, mounts
all project code and model/tokenizer data read-only, starts rank 1 remotely, and
runs the terminal UI on atlas5.

```bash
ATLAS_MODEL=/path/to/atlas-stage \
TOKENIZER=/path/to/tokenizer \
ARGO_MODEL=/path/on/argo3/to/checkpoint \
MAX_PROMPT_TOKENS=512 MAX_NEW_TOKENS=64 \
scripts/run_chat_cluster.sh
```

The model stages and tokenizer load once per process. Each user turn renders
the complete conversation with the tokenizer's real chat template and creates
fresh per-layer caches. `/reset` clears conversation history, `/quit` exits,
and EOF or Ctrl-C initiate shutdown. A conversation over the configured prompt
limit is rejected explicitly and is never truncated.

`ATLAS_MODEL`, `TOKENIZER`, and `ARGO_MODEL` are required. Override
`ARGO_HOST`, `REMOTE_CODE`, `REMOTE_LOG`, `ATLAS_ADDR`, `ARGO_ADDR`,
`GLOO_IFACE`, `PORT`, or the token limits when the tested defaults do not match
the deployment. The stable Docker image is only inspected and run; it is never
built or tagged.
