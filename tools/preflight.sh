#!/bin/bash
# preflight.sh [--need-gib N] [--allow-own] — before any GPU work: access, GPU/containers idle, weights, work dir, CX7,
# images. --allow-own: this project's containers may be running (parallel tests) — skips the GPU-idle/memory checks,
# still stops on llama-swap or any foreign container. Runs on the Mac. Any failure stops at once. Read-only everywhere.
set -uo pipefail
cd "$(dirname "$0")/.." && . tools/qlog.sh && . tools/dev.sh
NEED=100; OWN=0
while [ $# -gt 0 ]; do case $1 in --need-gib) NEED=$2; shift 2;; --allow-own) OWN=1; shift;; *) shift;; esac; done
Q=$MODELS/Qwen3.8-Flash-Next-FP4-Local

# one probe per node, key=value lines
NODE='
echo host=$(hostname)
echo cont=$(docker ps --format "{{.Names}}:{{.Label \"project\"}}" | tr "\n" " ")
echo apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr "\n" " " | tr -d ",")
echo util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
echo mem=$(awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo)
echo idx=$(md5sum < '$Q'/model.safetensors.index.json 2>/dev/null | cut -c1-8)
echo cfg=$(md5sum < '$Q'/config.json 2>/dev/null | cut -c1-8)
echo shards=$(ls '$Q'/tensors/*.safetensors 2>/dev/null | wc -l)
echo sbytes=$(stat -c %s '$Q'/tensors/*.safetensors 2>/dev/null | awk "{s+=\$1} END{print s+0}")
echo work=$([ -w '$WORK' ] && echo ok || echo missing)
echo disk=$(df -BG --output=avail '$WORK' 2>/dev/null | tail -1 | tr -dc 0-9)
echo base=$(docker images -q nvcr.io/nvidia/pytorch:26.07-py3 | cut -c1-12)
'
CX='for ip in '"${WORKER_CX[*]}"'; do ping -c1 -W1 -M do -s 8972 $ip >/dev/null 2>&1 && echo "cx_$ip=ok" || echo "cx_$ip=fail"; done'
PROBE="
  echo A_host=\$(hostname); echo A_swap=\$(if [ -n '$JUMP_GUARD_UNIT' ]; then systemctl is-active $JUMP_GUARD_UNIT; else echo inactive; fi)
  ssh -n -o ConnectTimeout=8 -o BatchMode=yes $HEAD $(printf %q "$NODE$CX") | sed 's/^/C_/'
  ssh -n -o ConnectTimeout=8 -o BatchMode=yes $WORKER $(printf %q "$NODE") | sed 's/^/D_/'
"
if [ -n "$JUMP" ]; then R=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$JUMP" "$PROBE" 2>/dev/null); else R=$(bash -c "$PROBE" 2>/dev/null); fi
v() { grep -m1 "^$1=" <<< "$R" | cut -d= -f2-; }

[ -z "$JUMP" ] || [ "$(v A_host)" = "$JUMP_HOSTNAME" ] || bad "pre access" "A unreachable"
[ "$(v C_host)" = "$HEAD_HOSTNAME" ] || bad "pre access" "C unreachable (through A)"
[ "$(v D_host)" = "$WORKER_HOSTNAME" ] || bad "pre access" "D unreachable (through A)"
ok "pre access" "A · C · D"
s=$(v A_swap); [ "$s" = inactive ] || bad "pre llama-swap" "$JUMP_GUARD_UNIT ${s:-?} — must be inactive"
ok "pre llama-swap" "${JUMP_GUARD_UNIT:-no guard unit} inactive"
for n in C D; do
  c=$(v ${n}_cont); f=$(tr ' ' '\n' <<< "$c" | grep . | grep -v ':qwen38-engine$' | cut -d: -f1 | paste -sd, -)
  o=$(tr ' ' '\n' <<< "$c" | grep ':qwen38-engine$' | cut -d: -f1 | paste -sd, -)
  [ -z "$f" ] || bad "pre containers $n" "foreign containers running: $f"
  [ -z "$o" ] || [ $OWN = 1 ] || bad "pre containers $n" "this project's containers still running: $o"
  [ $OWN = 1 ] && { ok "pre GPU $n" "no foreign containers · this project ${o:-none}"; continue; }
  a=$(v ${n}_apps); u=$(v ${n}_util); m=$(v ${n}_mem)
  [ -z "$a" ] || bad "pre GPU $n" "GPU processes running: $a"
  [ "${u:-100}" -lt 5 ] || bad "pre GPU $n" "utilization ${u}%"
  [ "${m:-0}" -ge "$NEED" ] || bad "pre GPU $n" "free memory ${m} GiB (need ${NEED})"
  ok "pre GPU $n" "idle · utilization ${u}% · ${m} GiB free"
done
[ "$(v C_shards)" = 21 ] && [ "$(v D_shards)" = 21 ] \
  || bad "pre weights" "shard count C $(v C_shards) / D $(v D_shards) (21)"
[ "$(v C_idx)" = "$(v D_idx)" ] && [ -n "$(v C_idx)" ] && [ "$(v C_cfg)" = "$(v D_cfg)" ] && [ "$(v C_sbytes)" = "$(v D_sbytes)" ] \
  || bad "pre weights" "C·D differ · index $(v C_idx)/$(v D_idx) · config $(v C_cfg)/$(v D_cfg) ·\
 bytes $(v C_sbytes)/$(v D_sbytes)"
ok "pre weights" "21 shards · $(( $(v C_sbytes) / 1073741824 )) GiB · index $(v C_idx) · config $(v C_cfg) · C=D"
for n in C D; do
  [ "$(v ${n}_work)" = ok ] || bad "pre work dir $n" "$WORK missing or not writable"
  [ "$(v ${n}_disk)" -ge 300 ] || bad "pre work dir $n" "disk free $(v ${n}_disk) GB (need 300)"
done
ok "pre work dir" "C $(v C_disk) GB · D $(v D_disk) GB free"
for ip in "${WORKER_CX[@]}"; do
  [ "$(v C_cx_$ip)" = ok ] || bad "pre CX7" "C→D jumbo ping failed ($ip $(v C_cx_$ip))"
done
ok "pre CX7" "C→D jumbo ping OK on both rails"
[ -n "$(v C_base)" ] || bad "pre image" "nvcr.io/nvidia/pytorch:26.07-py3 missing on C"
ok "pre image" "base $(v C_base) on C"
