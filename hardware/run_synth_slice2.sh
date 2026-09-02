#!/bin/bash
ulimit -c 0
echo core > /proc/sys/kernel/core_pattern
cd /root/jphys
for cfg in "32 8 4 141" "64 2 8 141" "32 8 4 64" "64 2 8 64"; do
  set -- $cfg
  T=stat_t1s$1n$2bw$4.txt
  if [ -f /root/jphys/out/$T ]; then echo "skip $T"; continue; fi
  echo "=== synth t1 D=$1 NE=$2 ACC=$3 BW=$4 ==="
  /usr/bin/time -v bash synth_slice2.sh "$1" "$2" "$3" "$4" 2>&1 | grep -E "DONE|FAIL|Maximum resident|Elapsed" || true
done
echo ALL_T1_SYNTH_DONE
