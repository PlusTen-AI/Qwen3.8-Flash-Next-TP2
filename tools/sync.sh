#!/bin/bash
# sync.sh — copy the repo (Mac) to <work>/src on the head, then head → worker over CX7. rsync --delete inside src only.
set -euo pipefail
cd "$(dirname "$0")/.." && . tools/qlog.sh && . tools/dev.sh
X=(--exclude .git --exclude .claude --exclude .venv --exclude '__pycache__' --exclude .DS_Store --exclude results/raw --exclude build)
RSH="ssh${JUMP:+ $JUMP ssh}"
rsync -a --delete "${X[@]}" -e "$RSH" ./ "$HEAD:$WORK/src/" || bad sync "copy to C failed"
on C "rsync -a --delete -e 'ssh -o BatchMode=yes' $WORK/src/ ${WORKER_CX[0]}:$WORK/src/" || bad sync "copy C→D failed"
n=$(on D "find $WORK/src -type f | wc -l")
rev=$(git rev-parse --short HEAD 2>/dev/null || echo no-commit)$(git diff --quiet HEAD 2>/dev/null || echo ' +modified')
ok sync "C·D src · ${n} files · $rev"
