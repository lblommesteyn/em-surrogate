# OpenROAD P&R for one matched domain netlist on Nangate45.
# Identical script for both domains; invoke with env vars TAG set.
# Flow: floorplan -> global place -> repair_design (buffering/sizing) ->
# detailed place -> CTS -> global route -> parasitics -> reports.
# repair_timing is deliberately omitted (identically for both): at the
# shared 2.0 ns constraint the unpipelined arrays are unclosable and it
# would only add unbounded runtime; achieved clock = reported arrival.
set tag $::env(TAG)
set out /root/jphys/out
set pdir /root/OpenROAD-flow-scripts/flow/platforms/nangate45
read_lef $pdir/lef/NangateOpenCellLibrary.tech.lef
read_lef $pdir/lef/NangateOpenCellLibrary.macro.mod.lef
read_liberty $pdir/lib/NangateOpenCellLibrary_typical.lib
read_verilog $out/net_$tag.v
link_design mac_domain
create_clock -name clk -period 2.0 [get_ports clk]
set_input_delay 0.2 -clock clk [all_inputs]
set_output_delay 0.2 -clock clk [all_outputs]
initialize_floorplan -utilization 60 -aspect_ratio 1 -core_space 4 \
  -site FreePDK45_38x28_10R_NP_162NW_34O
source $pdir/make_tracks.tcl
place_pins -hor_layers metal5 -ver_layers metal6
global_placement -density 0.72
estimate_parasitics -placement
repair_design
detailed_placement
report_design_area > $out/pnr_${tag}_area.txt
clock_tree_synthesis -buf_list BUF_X4 -sink_clustering_enable
set_propagated_clock [all_clocks]
detailed_placement
global_route -congestion_report_file $out/pnr_${tag}_congestion.rpt -verbose
estimate_parasitics -global_routing
report_checks -path_delay max -digits 3 > $out/pnr_${tag}_timing.txt
report_design_area >> $out/pnr_${tag}_area.txt
report_power > $out/pnr_${tag}_power.txt
puts "PNR_DONE $tag"
exit
