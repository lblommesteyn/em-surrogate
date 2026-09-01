# Re-run floorplan + gpl + macro_placement only, then dump DEF of macro
# positions for visualization. Identical settings to pnr_slice.tcl.
set tag $::env(TAG)
set out /root/jphys/out
set pdir /root/OpenROAD-flow-scripts/flow/platforms/nangate45
read_lef $pdir/lef/NangateOpenCellLibrary.tech.lef
read_lef $pdir/lef/NangateOpenCellLibrary.macro.mod.lef
read_lef $pdir/lef/fakeram45_128x256.lef
read_liberty $pdir/lib/NangateOpenCellLibrary_typical.lib
read_liberty $pdir/lib/fakeram45_128x256.lib
read_verilog $out/net_$tag.v
link_design mac_slice
create_clock -name clk -period 2.0 [get_ports clk]
initialize_floorplan -utilization 55 -aspect_ratio 1 -core_space 6 \
  -site FreePDK45_38x28_10R_NP_162NW_34O
source $pdir/make_tracks.tcl
place_pins -hor_layers metal5 -ver_layers metal6
global_placement -density 0.72 -init_density_penalty 0.001 -max_phi_coef 1.03
macro_placement -halo {4 4}
set f [open $out/macros_$tag.txt w]
foreach inst [get_cells -hierarchical *] {
  set i [sta::sta_to_db_inst $inst]
  if {$i != "NULL" && [[$i getMaster] isBlock]} {
    lassign [$i getLocation] x y
    puts $f "[$i getName] $x $y [$i getOrient]"
  }
}
close $f
puts "MACRO_DUMP_DONE $tag"
exit
