#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=spikingbrain-cpu:goal8.6-openblas
EXPECTED_IMAGE=sha256:696d98683cee6eef018a73e5c70b402f9103fc24b230356975a7ef1685aed836
ARGO_HOST=${ARGO_HOST:-argo3}
REMOTE_CODE=${REMOTE_CODE:-/tmp/spikingbrain-chat-runtime}
PORT=${PORT:-29614}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
ATLAS_ADDR=${ATLAS_ADDR:-192.168.1.128}
ARGO_ADDR=${ARGO_ADDR:-192.168.1.64}
GLOO_IFACE=${GLOO_IFACE:-eno1}
REMOTE_LOG=${REMOTE_LOG:-/tmp/spikingbrain-chat-argo.log}
CONTAINER=goal14-chat-argo

: "${ATLAS_MODEL:?set ATLAS_MODEL to the atlas5 stage model directory}"
: "${TOKENIZER:?set TOKENIZER to the tokenizer directory}"
: "${ARGO_MODEL:?set ARGO_MODEL to the argo3 model directory}"

test "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$EXPECTED_IMAGE"
test "$(ssh -n "$ARGO_HOST" docker image inspect "$IMAGE" --format '{{.Id}}')" = "$EXPECTED_IMAGE"

ssh -n "$ARGO_HOST" "mkdir -p '$REMOTE_CODE/src' '$REMOTE_CODE/scripts'"
rsync -a --delete "$ROOT/src/" "$ARGO_HOST:$REMOTE_CODE/src/"
rsync -a --delete \
  "$ROOT/scripts/distributed_generate.py" \
  "$ROOT/scripts/distributed_stage.py" \
  "$ROOT/scripts/persistent_chat.py" \
  "$ROOT/scripts/chat_argo.py" \
  "$ARGO_HOST:$REMOTE_CODE/scripts/"

cleanup() {
  ssh -n "$ARGO_HOST" "docker stop -t 10 '$CONTAINER' >/dev/null 2>&1 || true" || true
}
trap cleanup EXIT INT TERM

ssh -n "$ARGO_HOST" "nohup docker run --rm --name '$CONTAINER' --network host \
  -e GLOO_SOCKET_IFNAME='$GLOO_IFACE' \
  -v '$ARGO_MODEL:/model:ro' \
  -v '$REMOTE_CODE/src:/app/src:ro' \
  -v '$REMOTE_CODE/scripts:/app/scripts:ro' \
  '$IMAGE' python scripts/chat_argo.py \
  --master-addr '$ATLAS_ADDR' --port '$PORT' --peer '$ATLAS_ADDR' \
  --model-dir /model --max-prompt-tokens '$MAX_PROMPT_TOKENS' \
  --max-new-tokens '$MAX_NEW_TOKENS' --threads 4 --timeout 180 \
  >'$REMOTE_LOG' 2>&1 </dev/null &"

docker run --rm -i --name goal14-chat-atlas --network host \
  -e GLOO_SOCKET_IFNAME="$GLOO_IFACE" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "$ATLAS_MODEL:/model:ro" \
  -v "$TOKENIZER:/tokenizer:ro" \
  -v "$ROOT/src:/app/src:ro" \
  -v "$ROOT/scripts:/app/scripts:ro" \
  "$IMAGE" python scripts/chat_atlas.py \
  --master-addr "$ATLAS_ADDR" --port "$PORT" --peer "$ARGO_ADDR" \
  --model-dir /model --tokenizer-dir /tokenizer \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" --max-new-tokens "$MAX_NEW_TOKENS" \
  --threads 4 --timeout 180

ssh -n "$ARGO_HOST" "cat '$REMOTE_LOG'"
