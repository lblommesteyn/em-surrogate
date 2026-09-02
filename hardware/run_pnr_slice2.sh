#!/bin/bash
ulimit -c 0
echo core > /proc/sys/kernel/core_pattern
cd /root/jphys
while ! grep -q ALL_T1_SYNTH_DONE synth_t1.log 2>/dev/null; do sleep 120; done
for tag in t1s32n8bw141 t1s64n2bw141 t1s32n8bw64 t1s64n2bw64; do
  [ -f out/pnr_${tag}_timing.txt ] && { echo "skip $tag"; continue; }
  [ -f out/net_${tag}.v ] || { echo "no netlist $tag"; continue; }
  echo "=== PNR $tag ==="
  TAG=$tag /usr/bin/time -v openroad -no_init -exit pnr_slice2.tcl > out/pnr_${tag}.log 2>&1
  echo "exit=$? $tag"
done
echo ALL_T1_PNR_DONE
