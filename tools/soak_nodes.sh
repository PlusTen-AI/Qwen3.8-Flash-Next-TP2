#!/bin/bash
# soak_nodes.sh start ["C D"]|stop|fetch OUT|clean — (development, this site) node samples for tools/soak.py, read-only on the
# hosts: on C and D a loop writes every 2 s  epoch,MemAvailable kB,GPU use %,power W,SM clock MHz,temperature °C,
# clock-event reasons  to <node>.csv, and every 10 s  epoch,container,CPU %,memory  of this project's containers to
# <node>_docker.csv, in <work>/scratch/soak/. stop: the loops end within 2 s. fetch OUT: copies the
# files to OUT/nodes/. clean: removes the folder on both nodes.
set -uo pipefail
cd "$(dirname "$0")/.." && . tools/dev.sh
D=$WORK/scratch/soak
LOOP='n=$1; d=$2; mkdir -p "$d"; rm -f "$d/stop"; i=0
while [ ! -e "$d/stop" ]; do
  g=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits | head -1 | tr -d " ")
  echo "$(date +%s.%N | cut -c1-14),$(awk "/^MemAvailable/{print \$2}" /proc/meminfo),$g" >> "$d/$n.csv"
  if [ $((i % 5)) -eq 0 ]; then
    ( t=$(date +%s); docker stats --no-stream --format "{{.Name}},{{.CPUPerc}},{{.MemUsage}}" $(docker ps -q --filter label=project=qwen38-engine) 2>/dev/null | sed "s/^/$t,/" >> "$d/${n}_docker.csv" ) &
  fi
  i=$((i + 1)); sleep 2
done'
case ${1:-} in
  start)
    for n in ${2:-C D}; do
      on $n "mkdir -p $D && rm -f $D/$n.csv $D/${n}_docker.csv && printf '%s\n' $(printf %q "$LOOP") > $D/loop.sh; nohup bash $D/loop.sh $n $D </dev/null >/dev/null 2>&1 & sleep 3; tail -1 $D/$n.csv" \
        || { echo "soak_nodes: $n not started"; exit 1; }
    done ;;
  stop) for n in C D; do on $n "touch $D/stop"; done ;;
  fetch)
    O=${2:?fetch OUT}; mkdir -p "$O/nodes"
    for n in C D; do on $n "cat $D/$n.csv" > "$O/nodes/$n.csv"; on $n "cat $D/${n}_docker.csv 2>/dev/null" > "$O/nodes/${n}_docker.csv"; done
    wc -l "$O"/nodes/*.csv ;;
  clean) for n in C D; do on $n "rm -rf $D"; done ;;
  *) echo "usage: tools/soak_nodes.sh start|stop|fetch OUT|clean"; exit 1 ;;
esac
