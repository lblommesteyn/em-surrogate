#!/bin/bash
cd /root/jphys/out
for t in s32n8bw64 s64n2bw64 s32n8bw128 s64n2bw128; do
  [ -f pnr_${t}_timing.txt ] || continue
  echo "===== $t"
  grep -E "Design area" pnr_${t}.log | tail -1
  grep "Total wirelength" pnr_${t}.log
  echo "-- reg2reg:"
  grep -E "data arrival time|slack" pnr_${t}_timing.txt | tail -3
  grep -E "^Total" pnr_${t}_power.txt 2>/dev/null | tail -1
done
