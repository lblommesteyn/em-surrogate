// Parameterized weight-stationary MAC array with bandwidth-limited operand
// port, FIFO credit, backpressure, and performance counters.
// Same dataflow for every D (no per-size optimization):
//   for n_block: for m_block(<=ACC rows): {clear acc;
//       for k_chunk: LOADW (D rows, needs D bytes/row from FIFO)
//                    then stream m activation rows (D bytes/row, 1 MAC row/cycle)}
//   writeback is counted but uses a separate output port.
// int8 x int8 -> int32, one output ROW per firing cycle (D dot products of
// length D: D*D MACs/cycle, log2(D)-deep adder tree -> synthesis exposes the
// timing cost of wide arrays).
`timescale 1ns/1ps
module mac_array #(
    parameter D   = 32,
    parameter ACC = 8
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [15:0] cfg_m,      // rows this m_block (<= ACC)
    input  wire [15:0] cfg_kc,     // number of k chunks
    input  wire        acc_clear,
    // operand FIFO credit interface: feeder adds bytes, array consumes rows
    input  wire [31:0] fifo_bytes, // bytes currently available
    output wire [31:0] consume,    // bytes consumed this cycle (0 or D)
    output reg         done,
    // data (flattened row buses driven by TB in lockstep with consume)
    input  wire [8*D-1:0] in_row,  // weight row during LOADW, act row during RUN
    output wire [32*D-1:0] out_row0, // accumulator row 0 (checked by TB)
    // counters
    output reg [31:0] c_total, c_loadw, c_fire, c_starve
);
    localparam S_IDLE = 0, S_LOADW = 1, S_RUN = 2, S_DONE = 3;
    reg [1:0]  state;
    wire fire = (state == S_LOADW || state == S_RUN) && (fifo_bytes >= D);
    assign consume = fire ? D : 0;
    reg [15:0] wrow, mrow, kc;

    reg signed [7:0] w [0:D-1][0:D-1];     // w[k][n]
    reg signed [31:0] acc [0:ACC-1][0:D-1];
    integer i, j;

    // one output row of D dot products per firing cycle
    reg signed [31:0] dot [0:D-1];
    integer ci, cj;
    always @* begin
        // synthesis: full combinational dot; simulation guard (state==RUN)
        // avoids re-evaluating D*D multiplies during LOADW cycles where the
        // result is architecturally unused (no behavioral difference).
        if (state == S_RUN)
            for (cj = 0; cj < D; cj = cj + 1) begin
                dot[cj] = 0;
                for (ci = 0; ci < D; ci = ci + 1)
                    dot[cj] = dot[cj] +
                        $signed(in_row[8*ci +: 8]) * w[ci][cj];
            end
    end
    genvar g;
    generate for (g = 0; g < D; g = g + 1)
        assign out_row0[32*g +: 32] = acc[0][g];
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; done <= 0;
            c_total <= 0; c_loadw <= 0; c_fire <= 0; c_starve <= 0;
        end else begin
            if (state != S_IDLE && state != S_DONE) c_total <= c_total + 1;
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
                    for (j = 0; j < D; j = j + 1)
                        acc[mrow[2:0]][j] <= acc[mrow[2:0]][j] + dot[j];
                    c_fire <= c_fire + 1;
                    if (mrow == cfg_m - 1) begin
                        mrow <= 0;
                        if (kc == cfg_kc - 1) begin state <= S_DONE; done <= 1; end
                        else begin kc <= kc + 1; state <= S_LOADW; end
                    end else mrow <= mrow + 1;
                end else c_starve <= c_starve + 1;
            end
            S_DONE: ; // wait for rst/start
            endcase
        end
    end
endmodule
