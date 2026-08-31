// Behavioral simulation model of the fakeram45_128x256 macro (1RW port,
// 1-cycle read latency). Synthesis uses the liberty/LEF views instead.
`timescale 1ns/1ps
module fakeram45_128x256 (
    input  wire         clk,
    input  wire         ce_in,
    input  wire         we_in,
    input  wire [6:0]   addr_in,
    input  wire [255:0] wd_in,
    input  wire [255:0] w_mask_in,
    output reg  [255:0] rd_out
);
    reg [255:0] mem [0:127];
    always @(posedge clk) begin
        if (ce_in) begin
            if (we_in) mem[addr_in] <= (wd_in & w_mask_in) |
                                       (mem[addr_in] & ~w_mask_in);
            else rd_out <= mem[addr_in];
        end
    end
endmodule
