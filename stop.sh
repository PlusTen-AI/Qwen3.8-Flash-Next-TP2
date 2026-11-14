#!/bin/bash
# ./stop.sh — both ranks down (logs kept in <work>/results/serve/). Run on the head node.
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null \
  || { echo "[stop] needs python3 >= 3.11 on the host (for tomllib)"; exit 1; }
exec python3 "$(dirname "$0")/server/control.py" stop "$@"
