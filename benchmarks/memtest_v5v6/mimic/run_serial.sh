#!/bin/bash
# serial pestpp-ies run of mimic.pst, tracking the master's memory two ways.
#   run_serial.sh <exe> <tag> [cap_mb] [pst]
# every second writes: elapsed s, master rss mb, master physical footprint (top's MEM),
# model children rss mb.  kills the run if master rss goes over cap_mb (default 8000).
exe=$1; tag=$2; cap_mb=${3:-8000}; pst=${4:-mimic.pst}
here=$(cd "$(dirname "$0")" && pwd)
w=$here/run_$tag
rm -rf "$w"; mkdir -p "$w"
cp "$here"/$pst "$here"/mimic.*.csv "$here"/*.ins "$here"/model.in.tpl "$here"/forward_run.py \
   "$here"/prior_pe.bin "$here"/phi.csv "$w"/
(cd "$w" && exec "$exe" $pst > master.stdout 2>&1) &
mpid=$!
out=$here/${tag}.mem.csv
echo "t_s,master_rss_mb,master_footprint,children_rss_mb" > "$out"
t0=$(date +%s); peak=0; killed=0
while kill -0 $mpid 2>/dev/null; do
  mr=$(ps -o rss= -p $mpid 2>/dev/null | tr -d ' '); mr=${mr:-0}
  fp=$(top -l 1 -pid $mpid -stats mem 2>/dev/null | tail -1 | tr -d ' ')
  cr=0
  for c in $(pgrep -P $mpid 2>/dev/null); do
    r=$(ps -o rss= -p $c 2>/dev/null | tr -d ' '); cr=$((cr + ${r:-0}))
  done
  mmb=$((mr / 1024))
  echo "$(( $(date +%s) - t0 )),$mmb,$fp,$((cr / 1024))" >> "$out"
  [ $mmb -gt $peak ] && peak=$mmb
  if [ $mmb -gt $cap_mb ]; then
    echo "master over ${cap_mb} MB, killing"; pkill -P $mpid; kill $mpid; killed=1; break
  fi
  sleep 1
done
wait $mpid 2>/dev/null
fpk=$(cut -d, -f3 "$out" | tail -n +2 | sort -h | tail -1)
done_flag=$(grep -c "analysis complete" "$w/master.stdout")
echo "$tag: master peak rss ${peak} MB, peak footprint ${fpk}, killed=$killed, finished=$done_flag, wall $(( $(date +%s) - t0 )) s"
