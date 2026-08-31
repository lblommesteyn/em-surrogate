#!/bin/bash
cd /root/jphys/out
for t in d32n4bw64 d64n1bw64 d32n4bw128 d64n1bw128; do
  [ -f pnr_${t}_timing.txt ] || continue
  echo "===== $t"
  echo "area_file: $(cat pnr_${t}_area.txt 2>/dev/null)"
  grep -E "Design area" pnr_${t}.log | tail -1
  grep "Total wirelength" pnr_${t}.log
  echo "-- worst path (incl ports):"
  grep -E "data arrival time|slack" pnr_${t}_timing.txt | head -3
  echo "-- reg2reg:"
  grep -E "data arrival time|slack" pnr_${t}_timing.txt | tail -3
  echo "clkbuf_count: $(grep -oE "clkbuf[^ ]*" pnr_${t}.log | wc -l) cts_log: $(grep -E "Number of buffers|Clock net" pnr_${t}.log | head -2 | tr "\n" " ")"
  grep -E "^Total" pnr_${t}_power.txt | tail -1
  grep -E "congestion|Overflow report|overflow" pnr_${t}.log | tail -2
done
