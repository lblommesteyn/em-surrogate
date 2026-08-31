#!/bin/bash
cd /root/jphys
out=/root/jphys/slice_level2.txt
: > $out
for bw in 16 32 64 128 256 512; do
  iverilog -DDUT_D=32 -DDUT_N=8 -DDUT_ACC=4 -DDUT_BW=$bw -o tbs32_$bw tb_slice.v mac_slice.v mac_array.v fakeram45_128x256_beh.v
  ./tbs32_$bw +m=1 +kc=16 >> $out
  ./tbs32_$bw +m=4 +kc=16 >> $out
  iverilog -DDUT_D=64 -DDUT_N=2 -DDUT_ACC=8 -DDUT_BW=$bw -o tbs64_$bw tb_slice.v mac_slice.v mac_array.v fakeram45_128x256_beh.v
  ./tbs64_$bw +m=1 +kc=8 >> $out
  ./tbs64_$bw +m=8 +kc=8 >> $out
done
cat $out
