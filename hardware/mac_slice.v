// Matched Jalapeno-like compute SLICE for the granularity experiment.
//
//   mac_slice #(D=32, N_ENG=8, ACC=4, EXT_BW=..)   "A" (fine)
//   mac_slice #(D=64, N_ENG=2, ACC=8, EXT_BW=..)   "B" (coarse)
//
// Equal totals by construction: 8192 MACs, 1024 int32 accumulator words,
// 16 fakeram45_128x256 SRAM macros (64 KB) per slice. Each engine owns a
// bank of W = D/32 macros abreast (one D-byte row per access) x
// P = 16/(N_ENG*W) macros deep. One shared EXT_BW-byte/cycle external
// port; a round-robin distributor stages beats per engine and drains them
// into that engine's bank one row per cycle. Engines read rows through a
// two-entry skid buffer (1-cycle SRAM latency). Single-port macros: reads
// have priority, writes to the same depth layer stall. Every policy is
// parameter-generic and identical for both slices - the only difference
// is the partitioning. mac_array.v is reused verbatim.
`timescale 1ns/1ps
module mac_slice #(
    parameter D       = 32,
    parameter N_ENG   = 8,
    parameter ACC     = 4,
    parameter EXT_BW  = 64
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire [15:0]          cfg_m,
    input  wire [15:0]          cfg_kc,
    input  wire                 acc_clear,
    input  wire                 ext_valid,
    input  wire [8*EXT_BW-1:0]  ext_data,
    output wire                 ext_ready,
    output wire                 all_done,
    output wire [31:0]          obs,
    output wire [31:0]          c_total, c_starve_sum
);
    localparam ROWB    = D;                 // bytes per SRAM row read
    localparam W       = D/32;              // macros abreast
    localparam P       = 16/(N_ENG*W);      // macros deep per engine
    localparam CAPROWS = 128*P;             // rows per engine bank
    localparam PENDCAP = (2*EXT_BW > 2*ROWB) ? 2*EXT_BW : 2*ROWB;

    genvar e, wi, pi;
    wire [N_ENG-1:0] done_e;
    wire [N_ENG-1:0] pend_ok;
    wire [31:0] consume_e [0:N_ENG-1];
    wire [31:0] ctotal_e  [0:N_ENG-1];
    wire [31:0] cstarve_e [0:N_ENG-1];
    wire [31:0] obs_e     [0:N_ENG-1];

    // round-robin external distributor
    reg [3:0] rr;
    assign ext_ready = pend_ok[rr];
    wire beat = ext_valid && ext_ready;
    always @(posedge clk)
        if (rst) rr <= 0;
        else if (beat) rr <= (rr == N_ENG-1) ? 4'd0 : rr + 4'd1;

    generate for (e = 0; e < N_ENG; e = e + 1) begin : g_eng
        reg [31:0] pending;              // staged bytes not yet in SRAM
        reg [31:0] stored;               // rows resident in SRAM bank
        reg [9:0]  wrow, rrow;           // row pointers (CAPROWS <= 512)
        reg        rd_pend;              // read issued last cycle
        reg [2:0]  rd_layer_q;
        reg [1:0]  skid_cnt;
        reg [8*ROWB-1:0] skid0, skid1;
        wire [8*ROWB-1:0] rd_row;
        wire [32*D-1:0] out_row0_e;

        assign pend_ok[e] = (pending + EXT_BW <= PENDCAP);
        wire can_read  = (stored != 0) && ({1'b0, skid_cnt} + rd_pend < 2);
        wire [2:0] r_layer = rrow[9:7];  // depth layer = row / 128
        wire [2:0] w_layer = wrow[9:7];
        wire want_write = (pending >= ROWB) && (stored < CAPROWS);
        wire wr_go = want_write && !(can_read && (w_layer == r_layer));
        wire rd_go = can_read;
        wire [31:0] fifo_bytes_e = skid_cnt * ROWB;

        mac_array #(.D(D), .ACC(ACC)) uu (
            .clk(clk), .rst(rst), .start(start),
            .cfg_m(cfg_m), .cfg_kc(cfg_kc), .acc_clear(acc_clear),
            .fifo_bytes(fifo_bytes_e), .consume(consume_e[e]),
            .done(done_e[e]), .in_row(skid0[8*D-1:0]),
            .out_row0(out_row0_e), .c_total(ctotal_e[e]), .c_loadw(),
            .c_fire(), .c_starve(cstarve_e[e]));
        assign obs_e[e] = out_row0_e[31:0] ^ out_row0_e[32*D-1 -: 32];

        always @(posedge clk) begin
            if (rst) begin
                wrow <= 0; rrow <= 0; rd_pend <= 0; rd_layer_q <= 0;
                skid_cnt <= 0; skid0 <= 0; skid1 <= 0;
                pending <= 0; stored <= 0;
            end else begin
                pending <= pending
                    + ((beat && (e == rr)) ? EXT_BW : 0)
                    - (wr_go ? ROWB : 0);
                stored  <= stored + (wr_go ? 1 : 0) - (rd_go ? 1 : 0);
                if (wr_go) wrow <= (wrow == CAPROWS-1) ? 10'd0 : wrow + 10'd1;
                if (rd_go) begin
                    rrow <= (rrow == CAPROWS-1) ? 10'd0 : rrow + 10'd1;
                    rd_layer_q <= r_layer;
                end
                rd_pend <= rd_go;
                case ({rd_pend, (consume_e[e] != 0)})
                2'b10: begin
                    if (skid_cnt == 0) skid0 <= rd_row; else skid1 <= rd_row;
                    skid_cnt <= skid_cnt + 2'd1;
                end
                2'b01: begin
                    skid0 <= skid1; skid_cnt <= skid_cnt - 2'd1;
                end
                2'b11: begin
                    skid0 <= (skid_cnt == 1) ? rd_row : skid1;
                    if (skid_cnt == 2) skid1 <= rd_row;
                end
                default: ;
                endcase
            end
        end

        // ---- SRAM macros: W abreast x P deep ----
        wire [8*ROWB-1:0] rd_layer_data [0:P-1];
        for (pi = 0; pi < P; pi = pi + 1) begin : g_p
            wire lay_rd = rd_go && (r_layer == pi);
            wire lay_wr = wr_go && (w_layer == pi);
            wire [6:0] lay_addr = lay_wr ? wrow[6:0] : rrow[6:0];
            for (wi = 0; wi < W; wi = wi + 1) begin : g_w
                fakeram45_128x256 sram (
                    .clk(clk),
                    .ce_in(lay_rd | lay_wr),
                    .we_in(lay_wr),
                    .addr_in(lay_addr),
                    .wd_in({256{1'b1}}),
                    .w_mask_in({256{1'b1}}),
                    .rd_out(rd_layer_data[pi][256*wi +: 256]));
            end
        end
        assign rd_row = rd_layer_data[rd_layer_q >= P ? 0 : rd_layer_q];
    end endgenerate

    assign all_done = &done_e;

    integer k;
    reg [31:0] ct_r, cs_r, ob_r;
    always @* begin
        ct_r = 0; cs_r = 0; ob_r = 0;
        for (k = 0; k < N_ENG; k = k + 1) begin
            ct_r = ct_r + ctotal_e[k];
            cs_r = cs_r + cstarve_e[k];
            ob_r = ob_r ^ obs_e[k];
        end
    end
    assign obs = ob_r;
    assign c_total = ct_r;
    assign c_starve_sum = cs_r;
endmodule
