#!/bin/bash
ulimit -c 0
echo core > /proc/sys/kernel/core_pattern
cd /root/jphys
for tag in d32n4bw64 d64n1bw64 d32n4bw128 d64n1bw128; do
  [ -f out/pnr_${tag}_timing.txt ] && { echo "skip $tag"; continue; }
  echo "=== PNR $tag ==="
  TAG=$tag /usr/bin/time -v openroad -no_init -exit pnr_domain.tcl > out/pnr_${tag}.log 2>&1
  echo "exit=$? $tag"
done
echo ALL_PNR_DONE
