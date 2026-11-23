#!/bin/bash
# run one pestpp-ies master + one panther agent on a case and track the master's memory.
#   run_mem.sh <exe> <pst> <tag> [cap_mb]
# samples master and agent rss every 0.5 s into <tag>.mem.csv and kills both if the master
# goes over cap_mb (default 4000)
exe=$1; pst=$2; tag=$3; cap_mb=${4:-4000}
here=$(cd "$(dirname "$0")" && pwd)
port=$((4100 + RANDOM % 800))
m=$here/run_${tag}_master; a=$here/run_${tag}_agent
rm -rf "$m" "$a"; mkdir -p "$m" "$a"
for f in "$pst" model.in model.in.tpl model.out.ins forward_run.py external.*.csv; do
  cp "$here/$f" "$m/"; cp "$here/$f" "$a/"
done
(cd "$m" && exec "$exe" "$pst" /h :$port > master.stdout 2>&1) &
mpid=$!
sleep 1
(cd "$a" && exec "$exe" "$pst" /h localhost:$port > agent.stdout 2>&1) &
apid=$!
echo "t_s,master_rss_mb,agent_rss_mb" > "$here/${tag}.mem.csv"
t0=$(date +%s)
peak=0; killed=0
while kill -0 $mpid 2>/dev/null; do
  mr=$(ps -o rss= -p $mpid 2>/dev/null | tr -d ' '); mr=${mr:-0}
  ar=$(ps -o rss= -p $apid 2>/dev/null | tr -d ' '); ar=${ar:-0}
  mmb=$((mr / 1024)); amb=$((ar / 1024))
  echo "$(( $(date +%s) - t0 )),$mmb,$amb" >> "$here/${tag}.mem.csv"
  [ $mmb -gt $peak ] && peak=$mmb
  if [ $mmb -gt $cap_mb ]; then
    echo "master over ${cap_mb} MB, killing"; kill $mpid; killed=1; break
  fi
  sleep 0.5
done
wait $mpid 2>/dev/null
kill $apid 2>/dev/null; wait $apid 2>/dev/null
echo "$tag: master peak ${peak} MB, killed=$killed, wall $(( $(date +%s) - t0 )) s"
