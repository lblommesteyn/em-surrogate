// Tier-1 slice: pipelined engines (mac_array_p), MXFP4 feed accounting
// (row = D/2 bytes), drain counted. Bank packing is symmetric: one SRAM
// access moves TWO rows for both slices (fine: 2x16B rows in one 32B
// macro row; coarse: 2x32B rows across its 2-abreast macros), so total
// SRAM bytes stay equal and fully used. Same distributor, skid (4 rows),
// single-port read-priority arbitration as mac_slice.v - all policies
// parameter-generic and identical for both granularities.
`timescale 1ns/1ps
module mac_slice2 #(
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
    output wire [31:0]          c_total, c_starve_sum, c_drain_sum
);
    localparam ROWB    = D/2;               // bytes per row (4-bit operands)
    localparam W       = D/32;              // macros abreast
    localparam P       = 16/(N_ENG*W);      // macros deep per engine
    localparam CAPROWS = 2*128*P;           // rows per bank (2 rows/access)
    localparam PENDCAP = (2*EXT_BW > 4*ROWB) ? 2*EXT_BW : 4*ROWB;

    genvar e, wi, pi;
    wire [N_ENG-1:0] done_e;
    wire [N_ENG-1:0] pend_ok;
    wire [31:0] consume_e [0:N_ENG-1];
    wire [31:0] ctotal_e  [0:N_ENG-1];
    wire [31:0] cstarve_e [0:N_ENG-1];
    wire [31:0] cdrain_e  [0:N_ENG-1];
    wire [31:0] obs_e     [0:N_ENG-1];

    reg [3:0] rr;
    assign ext_ready = pend_ok[rr];
    wire beat = ext_valid && ext_ready;
    always @(posedge clk)
        if (rst) rr <= 0;
        else if (beat) rr <= (rr == N_ENG-1) ? 4'd0 : rr + 4'd1;

    generate for (e = 0; e < N_ENG; e = e + 1) begin : g_eng
        reg [31:0] pending;
        reg [31:0] stored;                   // rows resident in SRAM
        reg [10:0] wrow, rrow;               // row pointers (<= 1024 rows)
        reg        rd_pend;
        reg [2:0]  rd_layer_q;
        reg [2:0]  skid_cnt;                 // rows in skid (max 4)
        reg [8*ROWB-1:0] skid [0:3];
        wire [16*ROWB-1:0] rd_pair;          // one access = 2 rows
        wire [32*D-1:0] out_row0_e;

        assign pend_ok[e] = (pending + EXT_BW <= PENDCAP);
        wire can_read  = (stored >= 2) && (skid_cnt + (rd_pend?2:0) <= 2);
        wire [2:0] r_layer = rrow[10:8];     // 256 rows per macro layer
        wire [2:0] w_layer = wrow[10:8];
        wire want_write = (pending >= 2*ROWB) && (stored < CAPROWS-1);
        wire wr_go = want_write && !(can_read && (w_layer == r_layer));
        wire rd_go = can_read;
        wire [31:0] fifo_bytes_e = skid_cnt * ROWB;

        // engine consumes ROWB-byte rows; datapath in_row is 8D bits -
        // replicate the row (data is metric-dummy, timing is exact)
        wire [8*D-1:0] in_row_e = {skid[0], skid[0]};

        mac_array_p #(.D(D), .ACC(ACC), .ROWB(ROWB)) uu (
            .clk(clk), .rst(rst), .start(start),
            .cfg_m(cfg_m), .cfg_kc(cfg_kc), .acc_clear(acc_clear),
            .fifo_bytes(fifo_bytes_e), .consume(consume_e[e]),
            .done(done_e[e]), .in_row(in_row_e),
            .out_row0(out_row0_e), .c_total(ctotal_e[e]), .c_loadw(),
            .c_fire(), .c_starve(cstarve_e[e]), .c_drain(cdrain_e[e]));
        assign obs_e[e] = out_row0_e[31:0] ^ out_row0_e[32*D-1 -: 32];

        integer s;
        always @(posedge clk) begin
            if (rst) begin
                wrow <= 0; rrow <= 0; rd_pend <= 0; rd_layer_q <= 0;
                skid_cnt <= 0; pending <= 0; stored <= 0;
                for (s = 0; s < 4; s = s + 1) skid[s] <= 0;
            end else begin
                pending <= pending
                    + ((beat && (e == rr)) ? EXT_BW : 0)
                    - (wr_go ? 2*ROWB : 0);
                stored  <= stored + (wr_go ? 2 : 0) - (rd_go ? 2 : 0);
                if (wr_go) wrow <= (wrow >= CAPROWS-2) ? 11'd0 : wrow + 11'd2;
                if (rd_go) begin
                    rrow <= (rrow >= CAPROWS-2) ? 11'd0 : rrow + 11'd2;
                    rd_layer_q <= r_layer;
                end
                rd_pend <= rd_go;
                // skid: push 2 rows on read-return, pop 1 on consume
                case ({rd_pend, (consume_e[e] != 0)})
                2'b10: begin
                    skid[skid_cnt]   <= rd_pair[8*ROWB-1:0];
                    skid[skid_cnt+1] <= rd_pair[16*ROWB-1:8*ROWB];
                    skid_cnt <= skid_cnt + 3'd2;
                end
                2'b01: begin
                    for (s = 0; s < 3; s = s + 1) skid[s] <= skid[s+1];
                    skid_cnt <= skid_cnt - 3'd1;
                end
                2'b11: begin
                    for (s = 0; s < 3; s = s + 1) skid[s] <= skid[s+1];
                    skid[skid_cnt-1] <= rd_pair[8*ROWB-1:0];
                    skid[skid_cnt]   <= rd_pair[16*ROWB-1:8*ROWB];
                    skid_cnt <= skid_cnt + 3'd1;
                end
                default: ;
                endcase
            end
        end

        // SRAM macros: W abreast x P deep; one access = full width = 2 rows
        wire [256*W-1:0] rd_layer_data [0:P-1];
        for (pi = 0; pi < P; pi = pi + 1) begin : g_p
            wire lay_rd = rd_go && (r_layer == pi);
            wire lay_wr = wr_go && (w_layer == pi);
            wire [6:0] lay_addr = lay_wr ? wrow[7:1] : rrow[7:1];
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
        assign rd_pair = rd_layer_data[rd_layer_q >= P ? 0 : rd_layer_q][16*ROWB-1:0];
    end endgenerate

    assign all_done = &done_e;

    integer k;
    reg [31:0] ct_r, cs_r, cd_r, ob_r;
    always @* begin
        ct_r = 0; cs_r = 0; cd_r = 0; ob_r = 0;
        for (k = 0; k < N_ENG; k = k + 1) begin
            ct_r = ct_r + ctotal_e[k];
            cs_r = cs_r + cstarve_e[k];
            cd_r = cd_r + cdrain_e[k];
            ob_r = ob_r ^ obs_e[k];
        end
    end
    assign obs = ob_r;
    assign c_total = ct_r;
    assign c_starve_sum = cs_r;
    assign c_drain_sum = cd_r;
endmodule
