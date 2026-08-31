// Matched compute domain for the 32-vs-64 granularity experiment.
//
//   mac_domain #(D=32, N_UNITS=4, ACC=4, EXT_BW=..)   "A"
//   mac_domain #(D=64, N_UNITS=1, ACC=8, EXT_BW=..)   "B"
//
// Both have 4096 MACs, equal accumulator capacity (N_UNITS*ACC*D = 512
// int32 words), identical control, and one shared external operand port of
// EXT_BW bytes/cycle. A round-robin distributor forwards incoming beats to
// per-unit double row buffers; a unit fires when it holds >= D bytes. The
// distributor bus, fanout and muxing are the operand-delivery wiring whose
// physical cost the experiment measures - deliberately identical style for
// both sizes, no per-size optimization.
//
// The unit RTL (mac_array.v) is reused verbatim from jalapeno-sim.

`timescale 1ns/1ps
module mac_domain #(
    parameter D       = 32,
    parameter N_UNITS = 4,
    parameter ACC     = 4,
    parameter EXT_BW  = 64            // bytes per cycle on the external port
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   start,
    input  wire [15:0]            cfg_m,
    input  wire [15:0]            cfg_kc,
    input  wire                   acc_clear,
    // external operand port: one beat of EXT_BW bytes per cycle when valid
    input  wire                   ext_valid,
    input  wire [8*EXT_BW-1:0]    ext_data,
    output wire                   ext_ready,
    output wire                   all_done,
    // observability: OR-reduced accumulator row 0 of every unit (prevents
    // the tools from pruning the datapath; same reduction for both sizes)
    output wire [31:0]            obs,
    output wire [31:0]            c_total, c_starve_sum
);
    localparam BUF_BYTES = (2*D > 2*EXT_BW) ? 2*D : 2*EXT_BW; // double row buffer per unit (holds >= one beat)

    genvar u;
    wire [N_UNITS-1:0] done_u;
    wire [31:0] consume_u [0:N_UNITS-1];
    wire [31:0] ctotal_u  [0:N_UNITS-1];
    wire [31:0] cstarve_u [0:N_UNITS-1];
    reg  [31:0] fifo_bytes [0:N_UNITS-1];
    reg  [8*BUF_BYTES-1:0] rowbuf [0:N_UNITS-1];
    wire [8*D-1:0] in_row_u [0:N_UNITS-1];
    wire [32*D-1:0] out_row_u [0:N_UNITS-1];

    // round-robin distributor: each accepted beat goes to one unit's buffer
    reg [$clog2(N_UNITS+1)-1:0] rr;
    wire [31:0] free_bytes = BUF_BYTES - fifo_bytes[(N_UNITS==1)?0:rr];
    assign ext_ready = (free_bytes >= EXT_BW);
    wire beat = ext_valid && ext_ready;

    integer k;
    always @(posedge clk) begin
        if (rst) begin
            rr <= 0;
            for (k = 0; k < N_UNITS; k = k + 1) fifo_bytes[k] <= 0;
        end else begin
            for (k = 0; k < N_UNITS; k = k + 1) begin
                fifo_bytes[k] <= fifo_bytes[k]
                    + ((beat && (k == rr)) ? EXT_BW : 0)
                    - consume_u[k];
                if (beat && (k == rr))
                    rowbuf[k] <= (rowbuf[k] >> (8*EXT_BW))
                        | ({{(8*(BUF_BYTES-EXT_BW)){1'b0}}, ext_data}
                           << (8*(BUF_BYTES-EXT_BW)));
                else if (consume_u[k] != 0)
                    rowbuf[k] <= rowbuf[k] >> (8*D);
            end
            if (beat) rr <= (rr == N_UNITS-1) ? 0 : rr + 1;
        end
    end

    generate for (u = 0; u < N_UNITS; u = u + 1) begin : g_unit
        assign in_row_u[u] = rowbuf[u][8*D-1:0];
        mac_array #(.D(D), .ACC(ACC)) uu (
            .clk(clk), .rst(rst), .start(start),
            .cfg_m(cfg_m), .cfg_kc(cfg_kc), .acc_clear(acc_clear),
            .fifo_bytes(fifo_bytes[u]), .consume(consume_u[u]),
            .done(done_u[u]), .in_row(in_row_u[u]), .out_row0(out_row_u[u]),
            .c_total(ctotal_u[u]), .c_loadw(), .c_fire(),
            .c_starve(cstarve_u[u]));
    end endgenerate

    assign all_done = &done_u;

    // identical observability reduction for both sizes
    reg [31:0] obs_r;
    reg [31:0] ct_r, cs_r;
    always @* begin
        obs_r = 0; ct_r = 0; cs_r = 0;
        for (k = 0; k < N_UNITS; k = k + 1) begin
            ct_r = ct_r + ctotal_u[k];
            cs_r = cs_r + cstarve_u[k];
        end
    end
    integer m;
    reg [31:0] obs_acc;
    always @* begin
        obs_acc = 0;
        for (m = 0; m < N_UNITS; m = m + 1)
            obs_acc = obs_acc ^ out_row_u[m][31:0] ^ out_row_u[m][32*D-1 -: 32];
    end
    assign obs = obs_acc;
    assign c_total = ct_r;
    assign c_starve_sum = cs_r;
endmodule
