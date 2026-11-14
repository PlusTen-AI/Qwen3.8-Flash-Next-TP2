# Common shell logger — source it. One line per step: `[tag] point`. Green OK, red fail (exits), yellow warn.
# Long output goes to a file; the screen shows its path. NO_COLOR=1 turns colors off.
if [ -z "${NO_COLOR:-}" ]; then _QC=$'\033[36m'; _QG=$'\033[32m'; _QR=$'\033[31m'; _QY=$'\033[33m'; _Q0=$'\033[0m'
else _QC=; _QG=; _QR=; _QY=; _Q0=; fi
t()    { printf '%s[%s]%s %s\n' "$_QC" "$1" "$_Q0" "$2"; }                 # plain line
ok()   { printf '%s[%s]%s %s%s%s\n' "$_QC" "$1" "$_Q0" "$_QG" "$2" "$_Q0"; }
warn() { printf '%s[%s]%s %s%s%s\n' "$_QC" "$1" "$_Q0" "$_QY" "$2" "$_Q0"; }
bad()  { printf '%s[%s]%s %s%s%s\n' "$_QC" "$1" "$_Q0" "$_QR" "$2" "$_Q0"; exit 1; }
# chk TAG "test-command" "ok message" "fail message"  — fail exits
chk()  { if eval "$2" >/dev/null 2>&1; then ok "$1" "$3"; else bad "$1" "$4"; fi; }
