#!/bin/bash
# check_public.sh — (development) before a push: what git tracks must hold nothing of this site. Every tracked file is
# searched for tools/site.local.sh's PRIVATE pattern (addresses, host and user names), and the private files must not
# be tracked (CLAUDE.md, results/, serve.local.toml, tools/site.local.sh). Any hit fails. Korean text is counted (info).
set -uo pipefail
cd "$(dirname "$0")/.." && . tools/qlog.sh
[ -f tools/site.local.sh ] || bad public "tools/site.local.sh missing — its PRIVATE pattern is what this checks"
. tools/site.local.sh
[ -n "${PRIVATE:-}" ] || bad public "PRIVATE is empty in tools/site.local.sh"
T=$(git ls-files)
p=$(grep -E '^(CLAUDE\.md|serve\.local\.toml|tools/site\.local\.sh)$|^(results|\.work)/' <<< "$T" | head -5 | paste -sd' ' -)
[ -z "$p" ] || bad public "private files are tracked: $p"
hits=$(git ls-files -z | xargs -0 grep -niIE "$PRIVATE" 2>/dev/null | cut -c1-160 | head -20)
[ -z "$hits" ] || { echo "$hits"; bad public "this site's values in tracked files (above)"; }
ko=$(git ls-files -z | xargs -0 python3 -c 'import re,sys; print(sum(1 for f in sys.argv[1:]
  if re.search("[\uac00-\ud7a3]", open(f, errors="ignore").read())))' 2>/dev/null)
ok public "$(wc -l <<< "$T" | tr -d ' ') tracked files · nothing of this site · private files untracked · ${ko:-?} files with Korean text"
