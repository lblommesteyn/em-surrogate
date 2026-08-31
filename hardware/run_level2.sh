#!/bin/bash
# Level-2 domain cycle sweep: both domains x external-BW sweep.
# Rounds/composition are computed in the analysis; here we measure the
# canonical per-round configs (full accumulator occupancy and a fixed
# k-depth so both domains process identical bytes per unit-row).
cd ~/jphys
for BW in 16 32 64 128 256; do
  iverilog -g2005 -DDUT_D=32 -DDUT_N=4 -DDUT_ACC=4 -DDUT_BW=$BW -o s32_$BW.vvp mac_array.v mac_domain.v tb_domain.v
  iverilog -g2005 -DDUT_D=64 -DDUT_N=1 -DDUT_ACC=8 -DDUT_BW=$BW -o s64_$BW.vvp mac_array.v mac_domain.v tb_domain.v
  # equal-work rounds: K=512 -> kc32=16, kc64=8; batch cases m=1 (b1 decode)
  # and m=ACC (max occupancy)
  for M in 1 ACCMAX; do
    if [ "$M" = "ACCMAX" ]; then M32=4; M64=8; else M32=1; M64=1; fi
    vvp s32_$BW.vvp +m=$M32 +kc=16 | grep RESULT
    vvp s64_$BW.vvp +m=$M64 +kc=8  | grep RESULT
  done
done
