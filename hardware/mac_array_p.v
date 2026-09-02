// Tier-1 pipelined weight-stationary MAC array (evolves mac_array.v).
// Changes vs mac_array.v, identical for every D (no per-size tuning):
//  1. dot_col_p: registered after the multiply and after every 2 adder-tree
//     levels (staging RULE is the invariant; depth follows from D:
//     D=32 -> 3 regs, D=64 -> 4 regs; +accumulate ~ the 6-9 cycle
//     reduction the public record describes for the 64-wide unit).
//  2. ROWB parameter: bytes per streamed row (D/2 models MXFP4 operands
//     in FEED terms; the datapath remains int8 - disclosed conservatism).
//  3. DRAIN state: after the last accumulate retires, the ACC rows are
//     streamed out one row per cycle and counted (c_drain).
`timescale 1ns/1ps
module dot_col_p #(
    parameter D = 32
)(
    input  wire                clk,
    input  wire [8*D-1:0]      a,
    input  wire [8*D-1:0]      wcol,
    output wire signed [31:0]  y
);
    // stage 0: products, registered
    reg signed [15:0] p [0:D-1];
    integer i;
    always @(posedge clk)
        for (i = 0; i < D; i = i + 1)
            p[i] <= $signed(a[8*i +: 8]) * $signed(wcol[8*i +: 8]);

    // adder tree, register after every 2 levels
    generate
    if (D == 32) begin : g32
        reg signed [31:0] s2 [0:7];   // after levels 1-2
        reg signed [31:0] s4 [0:1];   // after levels 3-4
        reg signed [31:0] s5;         // final level
        integer j;
        always @(posedge clk) begin
            for (j = 0; j < 8; j = j + 1)
                s2[j] <= p[4*j] + p[4*j+1] + p[4*j+2] + p[4*j+3];
            for (j = 0; j < 2; j = j + 1)
                s4[j] <= s2[4*j] + s2[4*j+1] + s2[4*j+2] + s2[4*j+3];
            s5 <= s4[0] + s4[1];
        end
        assign y = s5;                // latency 4: p, s2, s4, s5
    end else begin : g64
        reg signed [31:0] s2 [0:15];
        reg signed [31:0] s4 [0:3];
        reg signed [31:0] s6;
        integer j;
        always @(posedge clk) begin
            for (j = 0; j < 16; j = j + 1)
                s2[j] <= p[4*j] + p[4*j+1] + p[4*j+2] + p[4*j+3];
            for (j = 0; j < 4; j = j + 1)
                s4[j] <= s2[4*j] + s2[4*j+1] + s2[4*j+2] + s2[4*j+3];
            s6 <= s4[0] + s4[1] + s4[2] + s4[3];
        end
        assign y = s6;                // latency 4: p, s2, s4, s6
    end
    endgenerate
endmodule

module mac_array_p #(
    parameter D    = 32,
    parameter ACC  = 8,
    parameter ROWB = 16               // bytes per streamed row (D/2 @4-bit)
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [15:0] cfg_m,
    input  wire [15:0] cfg_kc,
    input  wire        acc_clear,
    input  wire [31:0] fifo_bytes,
    output wire [31:0] consume,
    output reg         done,
    input  wire [8*D-1:0] in_row,
    output wire [32*D-1:0] out_row0,
    output reg  [31:0] c_total, c_loadw, c_fire, c_starve, c_drain
);
    localparam LAT = 4;               // dot_col_p latency (same rule both D)
    localparam S_IDLE=0, S_LOADW=1, S_RUN=2, S_FLUSH=3, S_DRAIN=4, S_DONE=5;
    reg [2:0] state;
    wire fire = (state == S_LOADW || state == S_RUN) && (fifo_bytes >= ROWB);
    assign consume = fire ? ROWB : 0;
    reg [15:0] wrow, mrow, kc, drow, fcnt;

    reg signed [7:0] w [0:D-1][0:D-1];
    reg signed [31:0] acc [0:ACC-1][0:D-1];
    integer i, j;

    // weight column buses into the pipelined dot columns
    wire signed [31:0] dot [0:D-1];
    genvar gc, gr;
    generate for (gc = 0; gc < D; gc = gc + 1) begin : g_col
        wire [8*D-1:0] wcol;
        for (gr = 0; gr < D; gr = gr + 1) begin : g_row
            assign wcol[8*gr +: 8] = w[gr][gc];
        end
        dot_col_p #(.D(D)) u_dot (.clk(clk), .a(in_row), .wcol(wcol), .y(dot[gc]));
    end endgenerate

    // in-flight tags: which acc row each pipeline slot retires into
    reg [LAT-1:0]      vpipe;
    reg [2:0]          mpipe [0:LAT-1];
    wire retire = vpipe[LAT-1];
    wire [2:0] mret = mpipe[LAT-1];

    generate for (gc = 0; gc < D; gc = gc + 1)
        assign out_row0[32*gc +: 32] = acc[0][gc];
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 0;
            c_total <= 0; c_loadw <= 0; c_fire <= 0; c_starve <= 0; c_drain <= 0;
            vpipe <= 0;
        end else begin
            if (state != S_IDLE && state != S_DONE) c_total <= c_total + 1;
            // pipeline shift + retire
            vpipe <= {vpipe[LAT-2:0], (state == S_RUN) && fire};
            mpipe[0] <= mrow[2:0];
            for (i = 1; i < LAT; i = i + 1) mpipe[i] <= mpipe[i-1];
            if (retire)
                for (j = 0; j < D; j = j + 1)
                    acc[mret][j] <= acc[mret][j] + dot[j];

            case (state)
            S_IDLE: if (start) begin
                state <= S_LOADW; wrow <= 0; mrow <= 0; kc <= 0; done <= 0;
                if (acc_clear)
                    for (i = 0; i < ACC; i = i + 1)
                        for (j = 0; j < D; j = j + 1) acc[i][j] <= 0;
            end
            S_LOADW: begin
                if (fire) begin
                    for (j = 0; j < D; j = j + 1)
                        w[wrow[$clog2(D)-1:0]][j] <= $signed(in_row[8*j +: 8]);
                    c_loadw <= c_loadw + 1;
                    if (wrow == D - 1) begin state <= S_RUN; wrow <= 0; mrow <= 0; end
                    else wrow <= wrow + 1;
                end else c_starve <= c_starve + 1;
            end
            S_RUN: begin
                if (fire) begin
                    c_fire <= c_fire + 1;
                    if (mrow == cfg_m - 1) begin
                        mrow <= 0;
                        if (kc == cfg_kc - 1) begin state <= S_FLUSH; fcnt <= 0; end
                        else begin kc <= kc + 1; state <= S_LOADW; end
                    end else mrow <= mrow + 1;
                end else c_starve <= c_starve + 1;
            end
            S_FLUSH: begin                    // wait for pipeline to retire
                fcnt <= fcnt + 1;
                if (fcnt == LAT) begin state <= S_DRAIN; drow <= 0; end
            end
            S_DRAIN: begin                    // stream ACC rows out, 1/cycle
                c_drain <= c_drain + 1;
                if (drow == cfg_m - 1) begin state <= S_DONE; done <= 1; end
                else drow <= drow + 1;
            end
            S_DONE: ;
            endcase
        end
    end
endmodule
