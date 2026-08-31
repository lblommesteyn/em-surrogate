#!/bin/bash
ulimit -c 0
echo core > /proc/sys/kernel/core_pattern
cd /root/jphys
for tag in s32n8bw64 s64n2bw64 s32n8bw128 s64n2bw128; do
  [ -f out/pnr_${tag}_timing.txt ] && { echo "skip $tag"; continue; }
  [ -f out/net_${tag}.v ] || { echo "no netlist $tag"; continue; }
  echo "=== PNR $tag ==="
  TAG=$tag /usr/bin/time -v openroad -no_init -exit pnr_slice.tcl > out/pnr_${tag}.log 2>&1
  echo "exit=$? $tag"
done
echo ALL_SLICE_PNR_DONE
