// Slice-level cycle testbench (same contract as tb_domain.v).
// -DDUT_D -DDUT_N (engines) -DDUT_ACC -DDUT_BW; +m= +kc=
`timescale 1ns/1ps
module tb_slice2;
    localparam D  = `DUT_D;
    localparam NE = `DUT_N;
    localparam AC = `DUT_ACC;
    localparam BW = `DUT_BW;

    reg clk = 0, rst = 1, start = 0;
    reg [15:0] cfg_m, cfg_kc;
    wire all_done, ext_ready;
    wire [31:0] obs, c_total, c_starve;
    reg  ext_valid;
    reg  [8*BW-1:0] ext_data;

    mac_slice2 #(.D(D), .N_ENG(NE), .ACC(AC), .EXT_BW(BW)) dut (
        .clk(clk), .rst(rst), .start(start), .cfg_m(cfg_m), .cfg_kc(cfg_kc),
        .acc_clear(1'b1), .ext_valid(ext_valid), .ext_data(ext_data),
        .ext_ready(ext_ready), .all_done(all_done), .obs(obs),
        .c_total(c_total), .c_starve_sum(c_starve));

    always #1 clk = ~clk;

    integer cycles, m_arg, kc_arg;
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
        $display("RESULT D=%0d NE=%0d ACC=%0d BW=%0d m=%0d kc=%0d cycles=%0d starve=%0d",
                 D, NE, AC, BW, m_arg, kc_arg, cycles, c_starve);
        $finish;
    end
endmodule
