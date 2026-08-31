#!/bin/bash
set -e
cd ~/jphys
LIB=~/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
run() {
  echo "=== synth D=$1 N=$2 ACC=$3 BW=$4 ==="
  /usr/bin/time -v bash synth_domain.sh "$1" "$2" "$3" "$4" "$LIB" ~/jphys/out 2>&1 \
    | grep -E "DONE|Maximum resident|Elapsed" || true
}
run 32 4 4 64
run 64 1 8 64
run 32 4 4 128
run 64 1 8 128
echo ALL_SYNTH_DONE
