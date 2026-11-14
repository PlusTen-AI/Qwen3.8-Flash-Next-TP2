#!/bin/bash
# run.sh [step...] — tools 1–5 in one go (Mac): sync → pre → image → table → dump → floor. Default: all, in that order.
# floor uses the newest dump (expert overlap measured from its router logits).
# Stops at the first failure. Screen: one line per step. Details: <work>/results/runs/<id>/ on the head (C),
# copied back (small files) to results/runs/<id>/ here. Raw dumps stay on C (rank0) and D (rank1).
set -uo pipefail
cd "$(dirname "$0")" && . tools/qlog.sh && . tools/dev.sh
STEPS=("${@:-all}"); [ "${STEPS[0]}" = all ] && STEPS=(sync pre image table dump floor)
ID=$(date +%Y%m%d-%H%M); W=$WORK; RUN=$W/results/runs/$ID
M=$MODELS
TAG=qwen38-engine/dev:$(shasum -a 256 docker/dev.Dockerfile | cut -c1-8)
c() { on C "$*"; }     # run on the head (C)
# dev <log name> <cmd...> — dev container on C (CPU unless GPU=1), project + weights (ro) mounted. Full output goes to
# $RUN/<log name>.log on C; the screen gets only the `[tag]` lines. On failure: the last error line + the log path.
dev() { local n=$1 g=""; shift; [ "${GPU:-0}" = 1 ] && g="--gpus all"
  c "docker run --rm --label project=qwen38-engine --user 1000:1000 -e HOME=/tmp -e HF_HOME=/tmp/hf $g -v $W:/work -v $M:/models:ro -w /work/src $TAG $* > $RUN/$n.log 2>&1; r=\$?; grep -aE '^(.\[36m)?\[[^]]+\]' $RUN/$n.log; [ \$r = 0 ] || { grep -aE 'Error|error' $RUN/$n.log | tail -1 | cut -c1-160; exit \$r; }"; }
t run "$ID · ${STEPS[*]}"
c "mkdir -p $RUN" || bad run "cannot reach C"

for s in "${STEPS[@]}"; do case $s in
sync)  tools/sync.sh || exit 1 ;;
pre)   tools/preflight.sh || exit 1 ;;
image)
  if [ -n "$(c "docker images -q $TAG")" ]; then ok image "$TAG present — build skipped"
  else
    c "cd $W/src/docker && docker build --label project=qwen38-engine -t $TAG -f dev.Dockerfile . > $RUN/image.log 2>&1" \
      || bad image "build failed · C $RUN/image.log"
    ok image "$TAG built · $(c "docker images $TAG --format '{{.ID}} {{.Size}}'")"
  fi
  v=$(c "docker run --rm --label project=qwen38-engine --user 1000:1000 -e HOME=/tmp --gpus all $TAG bash -c 'nvcc --version | grep -o \"release [0-9.]*\"; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader'" 2>&1 | paste -sd' ' -)
  grep -q 'GB10' <<< "$v" || bad image "GPU check failed · $v"
  ok image "CUDA ${v#release }" ;;
table)
  dev table python3 tools/tensor_table.py /models/Qwen3.8-Flash-Next-FP4-Local --out /work/results/tensor_table.tsv \
    || bad table "failed · C $RUN/table.log" ;;
floor)
  GPU=1 dev bw python3 tools/bw_probe.py /work/results/runs/$ID/bw.json \
    || bad floor "bandwidth probe failed · C $RUN/bw.log"
  dev floor python3 tools/byte_floor.py --config /models/Qwen3.8-Flash-Next-FP4-Local/config.json \
    --table /work/results/tensor_table.tsv --bw /work/results/runs/$ID/bw.json --out /work/results/runs/$ID/floor \
    $(c "ls -d $W/results/raw/dump/*/rank0 2>/dev/null | tail -1" | sed "s#^$W#--dump /work#") \
    || bad floor "failed · C $RUN/floor.log" ;;
dump)  c "bash $W/src/tools/dump/run_dump.sh $ID" || bad dump "failed · C $W/results/raw/dump/$ID/logs/" ;;
*) bad run "unknown step: $s" ;;
esac; done

mkdir -p results/runs/$ID
on C "cd $RUN && tar cf - --exclude=*.tsv . 2>/dev/null" | tar -C results/runs/$ID -xf - 2>/dev/null
ok run "done · $(( SECONDS / 60 )) min $(( SECONDS % 60 )) s · results/runs/$ID/"
