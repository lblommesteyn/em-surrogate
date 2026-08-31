#!/bin/bash
# Yosys synthesis + OpenSTA for one SLICE config on Nangate45 + fakeram.
# Usage: synth_slice.sh <D> <N_ENG> <ACC> <EXT_BW>
ulimit -c 0
D=$1; NE=$2; AC=$3; BW=$4
LIB=/root/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
RLIB=/root/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/fakeram45_128x256.lib
OUT=/root/jphys/out
mkdir -p $OUT
TAG=s${D}n${NE}bw${BW}
/root/oss-cad-suite/bin/yosys -q -p "
read_liberty -lib $RLIB;
read_verilog mac_array.v mac_slice.v;
chparam -set D $D -set N_ENG $NE -set ACC $AC -set EXT_BW $BW mac_slice;
hierarchy -top mac_slice;
rename -top mac_slice;
synth;
dfflibmap -liberty $LIB;
abc -fast -liberty $LIB;
opt_clean;
tee -o $OUT/stat_$TAG.txt stat -liberty $LIB;
write_verilog -noattr $OUT/net_$TAG.v" > $OUT/yosys_$TAG.log 2>&1
echo "YOSYS_EXIT=$?" >> $OUT/yosys_$TAG.log
[ -f $OUT/net_$TAG.v ] || { echo "FAIL $TAG"; exit 1; }
cat > $OUT/sta_$TAG.tcl <<TCL
read_liberty $LIB
read_liberty $RLIB
read_verilog $OUT/net_$TAG.v
link_design mac_slice
create_clock -name clk -period 2.0 [get_ports clk]
set_input_delay 0.2 -clock clk [all_inputs]
set_output_delay 0.2 -clock clk [all_outputs]
report_checks -path_delay max -digits 3 > $OUT/sta_$TAG.txt
exit
TCL
openroad -no_init -exit $OUT/sta_$TAG.tcl > $OUT/sta_log_$TAG.txt 2>&1 || true
echo "DONE $TAG"
