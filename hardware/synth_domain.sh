#!/bin/bash
# Yosys synthesis + OpenSTA timing for one domain config on Nangate45.
# Usage: synth_domain.sh <D> <N_UNITS> <ACC> <EXT_BW> <LIB> <OUTDIR>
D=$1; NU=$2; AC=$3; BW=$4; LIB=$5; OUT=$6
mkdir -p "$OUT"
TAG=d${D}n${NU}bw${BW}
yosys -q -p "
read_verilog mac_array.v mac_domain.v;
chparam -set D $D -set N_UNITS $NU -set ACC $AC -set EXT_BW $BW mac_domain;
hierarchy -top mac_domain;
synth -top mac_domain;
dfflibmap -liberty $LIB;
abc -fast -liberty $LIB;
opt_clean;
tee -o $OUT/stat_$TAG.txt stat -liberty $LIB;
write_verilog -noattr $OUT/net_$TAG.v" > $OUT/yosys_$TAG.log 2>&1
echo "YOSYS_EXIT=$?" >> $OUT/yosys_$TAG.log
[ -f $OUT/net_$TAG.v ] || { echo "FAIL $TAG"; exit 1; }
cat > $OUT/sta_$TAG.tcl <<TCL
read_liberty $LIB
read_verilog $OUT/net_$TAG.v
link_design mac_domain
create_clock -name clk -period 2.0 [get_ports clk]
set_input_delay 0.2 -clock clk [all_inputs]
set_output_delay 0.2 -clock clk [all_outputs]
report_checks -path_delay max -digits 3 > $OUT/sta_$TAG.txt
report_power >> $OUT/sta_$TAG.txt 2>/dev/null
exit
TCL
openroad -no_init -exit $OUT/sta_$TAG.tcl > $OUT/sta_log_$TAG.txt 2>&1 || sta $OUT/sta_$TAG.tcl > $OUT/sta_log_$TAG.txt 2>&1 || true
echo "DONE $TAG"
