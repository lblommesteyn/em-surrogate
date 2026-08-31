#!/bin/bash
cd /root/jphys
LIB=/root/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
for cfg in "64 1 8 64" "32 4 4 64" "32 4 4 128" "64 1 8 128"; do
  set -- $cfg
  T=stat_d$1n$2bw$4.txt
  if [ -f /root/jphys/out/$T ]; then echo "skip $T"; continue; fi
  echo "=== synth D=$1 N=$2 ACC=$3 BW=$4 ==="
  /usr/bin/time -v bash synth_domain.sh "$1" "$2" "$3" "$4" "$LIB" /root/jphys/out 2>&1 | grep -E "DONE|Maximum resident|Elapsed" || true
done
echo ALL_REST_DONE
