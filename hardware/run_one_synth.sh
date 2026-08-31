#!/bin/bash
cd /root/jphys
LIB=/root/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
echo "=== synth D=$1 N=$2 ACC=$3 BW=$4 ==="
/usr/bin/time -v bash synth_domain.sh "$1" "$2" "$3" "$4" "$LIB" /root/jphys/out 2>&1 | grep -E "DONE|Maximum resident|Elapsed" || true
