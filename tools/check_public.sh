#!/bin/bash
# check_public.sh — (development) before a push: what git tracks must hold nothing of this site. Every tracked file is
# searched for tools/site.local.sh's PRIVATE pattern (addresses, host and user names), and the private files must not
# be tracked (CLAUDE.md, results/, serve.local.toml, tools/site.local.sh). The history's commit messages and authors are
# searched too (the same pattern, e-mail addresses, session links); the only addresses let through are the author's
# GitHub noreply address and the co-author line's noreply@anthropic.com. Any hit fails. Korean text is counted (info).
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
msg=$(git log --format='%an <%ae> · %cn <%ce>%n%B' HEAD \
  | sed -E 's/<(260596924\+PlusTen-AI@users\.noreply\.github\.com|noreply@anthropic\.com)>/<>/g' \
  | grep -niE "$PRIVATE|claude\.ai/|session_[A-Za-z0-9]{8,}|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]{2,}" | cut -c1-160 | head -20)
[ -z "$msg" ] || { echo "$msg"; bad public "personal or site values in commit messages or authors (above)"; }
ko=$(git ls-files -z | xargs -0 python3 -c 'import re,sys; print(sum(1 for f in sys.argv[1:]
  if re.search("[\uac00-\ud7a3]", open(f, errors="ignore").read())))' 2>/dev/null)
ok public "$(wc -l <<< "$T" | tr -d ' ') tracked files · nothing of this site · private files untracked · commit messages and authors clean · ${ko:-?} files with Korean text"
