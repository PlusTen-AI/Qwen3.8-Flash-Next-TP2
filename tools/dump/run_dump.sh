#!/bin/bash
# run_dump.sh <run id> — golden dump (reference image: vLLM 0.32, TP=2 C–D, enforce-eager, no MTP, no prefix cache).
# Needs the reference deployment's vLLM image (site.local.sh DUMP_IMAGE) — removed with that recipe 2026-11-14; the
# dumps it made (<work>/results/raw/dump/20261111-1632 and 20261113-1910 on both nodes) stay the project's golden.
# YARN=<factor> (default 4: the serving default, ramp 262,144 at max-model-len 32768; 1.5625 = the first golden).
# Text prompts (client.py) then image/video chat requests on media drawn by media.py (vision tower hooked too).
# Runs on C. Call it through ../../run.sh (which runs the preflight first). Containers: qwen38-engine-dump on C and D,
# always removed at the end. Output: <work>/results/raw/dump/<id>/ on C (rank0 + tokens) and D (rank1).
set -uo pipefail
. "$(dirname "$0")/../site.local.sh"  # this site's addresses (synced with the repo; git ignores it)
ID=${1:?run id}
YARN=${YARN:-4}; MPE=$(python3 -c "print(int(262144 * $YARN))")
W=$WORK; SRC=$W/src; OUT=$W/results/raw/dump/$ID; CACHE=$W/cache/ref-dump
MD=$MODELS
IMG=$DUMP_IMAGE; LOAD_FORMAT=$DUMP_LOAD_FORMAT  # the reference image and its own weight loader (site.local.sh)
NAME=qwen38-engine-dump; PORT=8890; MPORT=25100
D=${WORKER_CX[0]}                              # D over CX7 (control)
H_IF=$DUMP_H_IF; H_IC=$DUMP_H_IC; H_HCA=$DUMP_H_HCA
W_IF=$DUMP_W_IF; W_IC=$DUMP_W_IC; W_HCA=$DUMP_W_HCA
. "$SRC/tools/qlog.sh"
sd() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$D" "$@"; }
mkdir -p "$OUT/logs" "$CACHE"; sd "mkdir -p $OUT $CACHE" || bad "dump prep" "cannot reach D"

CLEANED=0
cleanup() {  # once: keep both ranks' logs, then remove the containers
  [ $CLEANED = 1 ] && return; CLEANED=1
  docker logs "$NAME" > "$OUT/logs/rank0.log" 2>&1; sd "docker logs $NAME" > "$OUT/logs/rank1.log" 2>&1
  docker rm -f "$NAME" >/dev/null 2>&1; sd "docker rm -f $NAME >/dev/null 2>&1"
}
trap cleanup EXIT

# memory gate (UMA: wait until both boxes have the room), then drop the shard files from the page cache (no root)
for i in $(seq 1 24); do
  mh=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); mw=$(sd "awk '/MemAvailable/{printf \"%d\",\$2/1048576}' /proc/meminfo")
  [ "$mh" -ge 100 ] && [ "${mw:-0}" -ge 100 ] && break; sleep 5
done
[ "$mh" -ge 100 ] && [ "${mw:-0}" -ge 100 ] || bad "dump memory" "free C ${mh} / D ${mw} GiB (need 100)"
EV="docker run --rm --label project=qwen38-engine --user 1000:1000 -v $MD:/m:ro --entrypoint bash $IMG -c 'find /m/Qwen3.8-Flash-Next-FP4-Local/tensors -name \"*.safetensors\" -exec dd if={} iflag=nocache count=0 status=none \;'"
bash -c "$EV" && sd "$EV" || warn "dump cache" "page cache drop failed — going on"
# FlashInfer autotune cache must match on both ranks (a one-sided cache deadlocks warm-up) → clear both
rm -rf "$CACHE/vllm-cache/flashinfer_autotune_cache"; sd "rm -rf $CACHE/vllm-cache/flashinfer_autotune_cache"
ok "dump memory" "free C ${mh} / D ${mw} GiB · page cache dropped"
# test media (drawn, nothing downloaded): same image → same bytes; sha256 go into tokens.json
docker run --rm --network none --label project=qwen38-engine --user 1000:1000 -e HOME=/tmp -v $SRC:/src:ro -v $OUT:/out \
  --entrypoint python3 $IMG /src/tools/dump/media.py /out/media > "$OUT/logs/media.log" 2>&1 \
  && ok "dump media" "$(ls $OUT/media | grep -vc sha256) files · $OUT/media" \
  || bad "dump media" "media.py failed · $OUT/logs/media.log"

# RoCE v2 IPv4 GID index of <hca> on <iface> (moves on link flap/reboot)
GID='h=$1; f=$2; p=/sys/class/infiniband/$h/ports/1; b=""; for i in $(ls $p/gids | sort -n); do g=$(cat $p/gids/$i 2>/dev/null); t=$(cat $p/gid_attrs/types/$i 2>/dev/null); n=$(cat $p/gid_attrs/ndevs/$i 2>/dev/null); case "$g" in 0000:0000:0000:0000:0000:ffff:*) case "$t" in *v2*) [ "$n" = "$f" ] && { echo $i; exit; }; [ -z "$b" ] && b=$i;; esac;; esac; done; echo $b'
G0=$(bash -c "$GID" _ $H_HCA $H_IF); G1=$(sd "bash -c '$GID' _ $W_HCA $W_IF")
[ -n "$G0" ] && [ -n "$G1" ] || bad "dump RoCE" "GID not found (C '${G0}' / D '${G1}')"

HF='{"text_config":{"max_position_embeddings":'$MPE',"rope_parameters":{"rope_type":"yarn","factor":'$YARN',"original_max_position_embeddings":262144,"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}}}'
compose() {  # compose <rank> <iface> <ic> <hca> <gid>
  # the reference image's own loader / kernel switches (site.local.sh DUMP_IMAGE_ENV, NAME=VALUE each); the golden ran
  # with: empty the CUDA cache while loading on, loader prepare pass off, Marlin CSF path on, CSF bf16 for the vision
  # tower off, MXFP8 A16 path off, PLE prefill checkpoint off, PLE replication off
  local e img_env=(); for e in ${DUMP_IMAGE_ENV[@]+"${DUMP_IMAGE_ENV[@]}"}; do img_env+=(-e "$e"); done
  local a=(docker run -d --name $NAME --label project=qwen38-engine --user 1000:1000 --gpus all --ipc=host --network host
    --cap-add SYS_PTRACE --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 --cpuset-cpus 5-9,15-19
    -v $MD:/models:ro -v $CACHE:/cache -v $SRC/tools/dump/plugin:/qdump:ro -v $OUT:/out
    -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=180
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a
    -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm-cache -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace
    -e VLLM_MARLIN_USE_ATOMIC_ADD=0 -e VLLM_ENABLE_ROCE_ALLREDUCE=0
    ${img_env[@]+"${img_env[@]}"}
    -e VLLM_DISABLED_KERNELS=FlashInferCutedslMxfp8LinearKernel,FlashInferTrtllmMxfp8LinearKernel,DeepGemmMxfp8BmmLinearKernel,HummingMxfp8LinearKernel
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e MALLOC_MMAP_THRESHOLD_=65536 -e MALLOC_TRIM_THRESHOLD_=131072
    -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$4 -e NCCL_IB_GID_INDEX=$5 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET
    -e NCCL_IB_QPS_PER_CONNECTION=1 -e NCCL_IB_SPLIT_DATA_ON_QPS=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_CUMEM_ENABLE=0
    -e NCCL_NVLS_ENABLE=0 -e NCCL_MAX_NCHANNELS=4 -e NCCL_SOCKET_IFNAME=$2 -e GLOO_SOCKET_IFNAME=$2 -e VLLM_HOST_IP=$3
    -e PYTHONPATH=/qdump -e QDUMP_DIR=/out -e QDUMP_ROWS=64 -e QDUMP_MAX_STEPS=200 -e QDUMP_TENSOR_MB=64
    -e QDUMP_SKIP=none -e QDUMP_MM_DECODE_STEPS=2 -e 'QDUMP_SKIP_RE=visual\.blocks\.([1-9]|1[0-9]|2[0-5])\.'
    -e "QDUMP_FULLROWS=fn\._forward_core#[0-9]+\.in\.(mixed_qkv|a|b)|self_attn\.qkv_proj#[0-9]+\.out\.0|fn\.embed_input_ids#[0-9]+\.(ret|in\.)"
    --entrypoint vllm $IMG serve /models/Qwen3.8-Flash-Next-FP4-Local --host 127.0.0.1 --port $PORT
    --nnodes 2 --node-rank $1 --master-addr $H_IC --master-port $MPORT --tensor-parallel-size 2)
  [ "$1" != 0 ] && a+=(--headless)
  a+=(--served-model-name qdump --enforce-eager --distributed-executor-backend mp --pipeline-parallel-size 1
    --load-format "$LOAD_FORMAT" --moe-backend marlin --kernel-config '{"linear_backend_per_quant":{"mxfp8":"b12x"}}'
    --engram-config '{"cpu_offload": false}' --mm-encoder-tp-mode data
    --gpu-memory-utilization 0.75 --kv-cache-memory-bytes 8000000000 --kv-cache-dtype auto --mamba-ssm-cache-dtype float32
    --max-model-len 32768 --hf-overrides "$HF" --block-size 3456 --max-num-seqs 1 --max-num-batched-tokens 16384
    --no-enable-prefix-caching)
  printf '%q ' "${a[@]}"
}
docker rm -f $NAME >/dev/null 2>&1; sd "docker rm -f $NAME >/dev/null 2>&1"
eval "$(compose 0 $H_IF $H_IC $H_HCA $G0)" >/dev/null || bad "dump boot" "rank0 docker run failed"
sleep 2
sd "$(compose 1 $W_IF $W_IC $W_HCA $G1) >/dev/null" || bad "dump boot" "rank1 docker run failed"
ok "dump boot" "rank0 C · rank1 D · GID $G0/$G1 · eager · ssm fp32 · prefix cache off · MTP off"

# wait healthy; print each boot milestone once (from rank0's log)
T0=$(date +%s); seen=""
ms() { local k=$1 pat=$2 msg=$3; case " $seen " in *" $k "*) return;; esac
  docker logs $NAME 2>&1 | grep -qE "$pat" && { seen="$seen $k"; t "dump boot" "$msg · $(( $(date +%s) - T0 ))s"; }; }
while :; do
  curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  ms plugin '\[qdump\] rank0' "dump hooks installed"
  ms weights 'Loading weights took|Model loading took' "weights loaded"
  ms kv 'Available KV cache memory|GPU KV cache size' "KV allocated"
  for side in C D; do
    if [ $side = C ]; then up=$(docker ps -q -f name=^$NAME$); else up=$(sd "docker ps -q -f name=^$NAME$"); fi
    [ -n "$up" ] || { cleanup; e=$(grep -hE 'Error|Traceback' "$OUT"/logs/rank0.log "$OUT"/logs/rank1.log | grep -vE 'should dump|TCPStore|^\s*$' | tail -1 | cut -c1-160)
      bad "dump boot" "$side container exited · ${e:-cause unknown} · logs $OUT/logs/"; }
  done
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && bad "dump boot" "not ready within 40 min · logs $OUT/logs/"
  sleep 10
done
ok "dump ready" "$(( $(date +%s) - T0 ))s · :$PORT"

docker run --rm --label project=qwen38-engine --user 1000:1000 -e HOME=/tmp --network host \
  -v $SRC:/src:ro -v $MD:/models:ro -v $OUT:/out --entrypoint python3 $IMG \
  /src/tools/dump/client.py http://127.0.0.1:$PORT /out /models/Qwen3.8-Flash-Next-FP4-Local --media /out/media \
  || bad "dump request" "client failed"

cleanup; trap - EXIT
# per-rank totals
r0=$(python3 -c "import json;L=[json.loads(x) for x in open('$OUT/rank0/manifest.jsonl')];print(len(L),round(sum(x['bytes'] for x in L)/2**30,2),sum(len(x['skipped']) for x in L))")
r1=$(sd "python3 -c \"import json;L=[json.loads(x) for x in open('$OUT/rank1/manifest.jsonl')];print(len(L),round(sum(x['bytes'] for x in L)/2**30,2),sum(len(x['skipped']) for x in L))\"")
set -- $r0; s0=$1; g0=$2; k0=$3; set -- $r1; s1=$1; g1=$2; k1=$3
[ "$s0" = "$s1" ] && [ "${s0:-0}" -gt 0 ] && K=ok || K=bad
$K "dump saved" "steps rank0 $s0 / rank1 $s1 · ${g0}+${g1} GiB · skipped $k0/$k1 · $OUT"
# rank1 → C over CX7 (containers are gone, GPU idle) so tests on C can check TP partial sums against both ranks
sd "rsync -a $OUT/rank1/ $HEAD_CX:$OUT/rank1/" && [ -f "$OUT/rank1/manifest.jsonl" ] \
  && ok "dump gather" "rank1 → C ($(du -sh "$OUT/rank1" | cut -f1)) · the D copy stays" \
  || warn "dump gather" "rank1 copy failed — only on D"
