#!/bin/bash
# dev.sh — (Mac, development, this site only — not part of the public recipe) reach the nodes and drive ./start.sh on
# the head. Sourced: defines `on C|D <cmd>` and HEAD_REPO. Run: tools/dev.sh start|restart|stop|status|logs [args]
# — start/restart run tools/preflight.sh (site guards: llama-swap, other containers, GPUs idle) and tools/sync.sh
# first. Engine switches set here (Q38_PATCHES, Q38_DRAFT_LM, …) are passed on.
_SITE="$(dirname "${BASH_SOURCE[0]}")/site.local.sh"  # this site's addresses (git ignores it; see site.local.sh.example)
[ -f "$_SITE" ] || { echo "$_SITE missing — copy tools/site.local.sh.example to tools/site.local.sh and fill it in" >&2; exit 1; }
. "$_SITE"
HEAD_REPO=$WORK/src
# on C|D <cmd>: run on the head (C) or the worker (D), through the jump host when there is one
on() { local h; case $1 in C) h=$HEAD;; D) h=$WORKER;; *) h=$1;; esac
  if [ -n "$JUMP" ]; then ssh -o BatchMode=yes "$JUMP" "ssh -o BatchMode=yes $h $(printf %q "$2")"; else ssh -o BatchMode=yes "$h" "$2"; fi; }
pass_env() { local v e=""; for v in Q38_COMMIT Q38_PATCHES Q38_RDMA_INIT Q38_DRAFT_LM Q38_RDMA_TIMEOUT_S CUDA_LAUNCH_BLOCKING NCCL_DEBUG Q38_TEST_KV_NOGROW; do
  [ -n "${!v:-}" ] && e="$e $v=$(printf %q "${!v}")"; done; echo "$e"; }
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -uo pipefail
  cd "$(dirname "$0")/.." && . tools/qlog.sh
  Q38_COMMIT="$(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD 2>/dev/null || echo ' +modified')"
  cmd=${1:-}; shift
  case "$cmd" in
    start) tools/preflight.sh ${Q38_PRE:-} || exit 1; tools/sync.sh || exit 1 ;;
    restart) tools/preflight.sh --allow-own || exit 1; tools/sync.sh || exit 1 ;;
    stop|status|logs) ;;
    *) bad dev "usage: tools/dev.sh start|restart|stop|status|logs [args]" ;;
  esac
  on C "cd $HEAD_REPO && env $(pass_env) ./$([ "$cmd" = stop ] && echo stop.sh || echo "start.sh $cmd") ${*:+$(printf '%q ' "$@")}"
fi
