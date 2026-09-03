# OpenROAD P&R for one SLICE netlist on Nangate45 + fakeram45 macros.
# Identical script for both granularities (env TAG selects netlist).
# Same discipline as pnr_domain.tcl: no repair_timing, achieved clock =
# reg-to-reg arrival after CTS + global routing parasitics.
set tag $::env(TAG)
set out /root/jphys/out
set pdir /root/OpenROAD-flow-scripts/flow/platforms/nangate45
read_lef $pdir/lef/NangateOpenCellLibrary.tech.lef
read_lef $pdir/lef/NangateOpenCellLibrary.macro.mod.lef
read_lef $pdir/lef/fakeram45_128x256.lef
read_liberty $pdir/lib/NangateOpenCellLibrary_typical.lib
read_liberty $pdir/lib/fakeram45_128x256.lib
read_verilog $out/net_$tag.v
link_design mac_slice2
create_clock -name clk -period 2.0 [get_ports clk]
set_input_delay 0.2 -clock clk [all_inputs]
set_output_delay 0.2 -clock clk [all_outputs]
initialize_floorplan -utilization 45 -aspect_ratio 1 -core_space 6 \
  -site FreePDK45_38x28_10R_NP_162NW_34O
source $pdir/make_tracks.tcl
place_pins -hor_layers metal5 -ver_layers metal6
global_placement -density 0.72 -init_density_penalty 0.001 -max_phi_coef 1.03
macro_placement -halo {4 4}
global_placement -density 0.72 -init_density_penalty 0.001 -max_phi_coef 1.03
estimate_parasitics -placement
repair_design
detailed_placement
clock_tree_synthesis -buf_list BUF_X4 -sink_clustering_enable
set_propagated_clock [all_clocks]
detailed_placement
estimate_parasitics -placement
report_checks -path_delay max -digits 3 > $out/pnr_${tag}_timing.txt
report_checks -path_delay max -from [all_registers] -to [all_registers] -digits 3 >> $out/pnr_${tag}_timing.txt
catch {
  set fa [open $out/pnr_${tag}_area.txt w]
  puts $fa [rsz::design_area]
  close $fa
}
report_design_area
report_power > $out/pnr_${tag}_power.txt
puts "PRE_ROUTE_METRICS_SAVED $tag"
set_routing_layers -signal metal2-metal5
global_route -congestion_report_file $out/pnr_${tag}_congestion.rpt -verbose
estimate_parasitics -global_routing
report_checks -path_delay max -from [all_registers] -to [all_registers] -digits 3 > $out/pnr_${tag}_timing_routed.txt
puts "PNR_DONE $tag"
exit
