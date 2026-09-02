#!/bin/bash
cd /root/jphys
out=/root/jphys/slice2_sweep.txt
: > $out
for bw in 9 32 64 141 256; do
  iverilog -DDUT_D=32 -DDUT_N=8 -DDUT_ACC=4 -DDUT_BW=$bw -o t2a_$bw tb_slice2.v mac_slice2.v mac_array_p.v fakeram45_128x256_beh.v
  ./t2a_$bw +m=1 +kc=16 >> $out
  ./t2a_$bw +m=4 +kc=16 >> $out
  iverilog -DDUT_D=64 -DDUT_N=2 -DDUT_ACC=8 -DDUT_BW=$bw -o t2b_$bw tb_slice2.v mac_slice2.v mac_array_p.v fakeram45_128x256_beh.v
  ./t2b_$bw +m=1 +kc=8 >> $out
  ./t2b_$bw +m=8 +kc=8 >> $out
done
cat $out
