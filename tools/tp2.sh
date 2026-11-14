#!/bin/bash
# tp2.sh <name> <cmd...> — (Mac, development) the same command on both ranks through server/control.py pair on the
# head: repo synced, link read at start, `--rank R` appended, the weights' parent folder mounted at /models, logs →
# <work>/results/runs/tp2-<name>-<time>/ on the head. Q38_DUAL_RAIL=0 keeps one rail; Q38_TP2_TIMEOUT (s, 1800).
set -uo pipefail
cd "$(dirname "$0")/.." && . tools/dev.sh
n=${1:?name}; shift
tools/sync.sh >/dev/null || exit 1
rails=""; [ "${Q38_DUAL_RAIL:-1}" = 0 ] && rails="--set cluster.rails=1"
on C "cd $HEAD_REPO && env $(pass_env) python3 server/control.py $rails pair $n --timeout ${Q38_TP2_TIMEOUT:-1800} --models-parent -- $(printf '%q ' "$@")"
