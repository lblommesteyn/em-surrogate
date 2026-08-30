// Domain-level cycle testbench: measures cycles for one "round" in which
// every unit in the domain executes cfg_m rows x cfg_kc k-chunks, fed by a
// single EXT_BW bytes/cycle external port through the round-robin
// distributor. Workload latency = rounds(shape) x measured round cycles
// (composition done in the analysis script; identical for both domains).
//
// Configure with -DDUT_D=.. -DDUT_N=.. -DDUT_ACC=.. -DDUT_BW=..
// Plusargs: +m=<rows per m_block> +kc=<k chunks>
`timescale 1ns/1ps
module tb_domain;
    localparam D  = `DUT_D;
    localparam NU = `DUT_N;
    localparam AC = `DUT_ACC;
    localparam BW = `DUT_BW;

    reg clk = 0, rst = 1, start = 0;
    reg [15:0] cfg_m, cfg_kc;
    wire all_done, ext_ready;
    wire [31:0] obs, c_total, c_starve;
    reg  ext_valid;
    reg  [8*BW-1:0] ext_data;

    mac_domain #(.D(D), .N_UNITS(NU), .ACC(AC), .EXT_BW(BW)) dut (
        .clk(clk), .rst(rst), .start(start), .cfg_m(cfg_m), .cfg_kc(cfg_kc),
        .acc_clear(1'b1), .ext_valid(ext_valid), .ext_data(ext_data),
        .ext_ready(ext_ready), .all_done(all_done), .obs(obs),
        .c_total(c_total), .c_starve_sum(c_starve));

    always #1 clk = ~clk;

    integer cycles;
    integer m_arg, kc_arg;
    initial begin
        if (!$value$plusargs("m=%d", m_arg)) m_arg = AC;
        if (!$value$plusargs("kc=%d", kc_arg)) kc_arg = 4;
        cfg_m = m_arg[15:0]; cfg_kc = kc_arg[15:0];
        ext_valid = 1; ext_data = {8*BW{1'b1}};
        repeat (4) @(posedge clk);
        rst = 0; @(posedge clk);
        start = 1; @(posedge clk); start = 0;
        cycles = 0;
        while (!all_done && cycles < 4000000) begin
            @(posedge clk); cycles = cycles + 1;
        end
        $display("RESULT D=%0d N=%0d ACC=%0d BW=%0d m=%0d kc=%0d cycles=%0d starve=%0d",
                 D, NU, AC, BW, m_arg, kc_arg, cycles, c_starve);
        $finish;
    end
endmodule
