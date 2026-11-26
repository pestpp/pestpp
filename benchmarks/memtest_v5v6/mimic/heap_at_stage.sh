#!/bin/bash
# run mimic.pst with malloc stack logging on, pause the master as soon as a given line shows
# up in mimic.log, dump what memory is held and by which code, then let it finish.
#   heap_at_stage.sh <exe> <tag> "<log text to stop at>"
# writes <tag>.heap.txt (by allocation type) and <tag>.tree.txt (live bytes by call tree)
exe=$1; tag=$2; stop_at=$3; pst=${4:-mimic.pst}
here=$(cd "$(dirname "$0")" && pwd)
w=$here/heap_$tag
rm -rf "$w"; mkdir -p "$w"
cp "$here"/$pst "$here"/mimic.*.csv "$here"/*.ins "$here"/model.in.tpl "$here"/forward_run.py \
   "$here"/prior_pe.bin "$here"/phi.csv "$w"/
(cd "$w" && MallocStackLogging=1 exec "$exe" $pst > master.stdout 2>&1) &
mpid=$!
until grep -q "$stop_at" "$w/${pst%.pst}.log" 2>/dev/null; do
  kill -0 $mpid 2>/dev/null || { echo "master exited before '$stop_at'"; exit 1; }
  sleep 0.05
done
kill -STOP $mpid
echo "paused at: $(tail -1 "$w/${pst%.pst}.log")"
echo "rss_mb $(( $(ps -o rss= -p $mpid) / 1024 ))  footprint $(top -l 1 -pid $mpid -stats mem | tail -1)"
heap -s $mpid > "$here/${tag}.heap.txt" 2>&1
malloc_history $mpid -callTree -invert > "$here/${tag}.tree.txt" 2>&1
kill -CONT $mpid
wait $mpid
echo "$tag done: heap $(wc -l < "$here/${tag}.heap.txt") lines, tree $(wc -l < "$here/${tag}.tree.txt") lines"
