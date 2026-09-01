#!/bin/bash
ulimit -c 0
cd /root/jphys
for tag in s32n8bw64 s64n2bw64; do
  [ -f out/macros_${tag}.txt ] && { echo "skip $tag"; continue; }
  echo "=== DUMP $tag ==="
  TAG=$tag openroad -no_init -exit dump_macro_def.tcl > out/dump_${tag}.log 2>&1
  echo "exit=$? $tag"
done
echo ALL_DUMPS_DONE
