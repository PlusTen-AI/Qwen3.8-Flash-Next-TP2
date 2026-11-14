#!/bin/bash
# ./start.sh [start] | restart | status | logs [N]   [--set section.key=value ...]
# Run on the head node. Settings: serve.toml + serve.local.toml (copy serve.local.toml.example). See README.md.
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null \
  || { echo "[start] needs python3 >= 3.11 on the host (for tomllib)"; exit 1; }
exec python3 "$(dirname "$0")/server/control.py" "$@"
