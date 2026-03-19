`default_nettype none

module memory (
 input wire clk,
 input wire rst,
 input wire load_en,
 input wire [2:0] addr, // MSB selects matrix (0: weights, 1: inputs), [1:0] selects element
 input wire [7:0] in_data, // 8-bit data written into sram
 output wire [7:0] weight0, weight1, weight2, weight3, // 2x2 weights (A) elements
 output wire [7:0] input0, input1, input2, input3 // 2x2 inputs (B) elements
);

  reg [7:0] sram [0:7]; // 8 locations: 0-3 for weights, 4-7 for inputs
  integer i;

  always @(posedge clk) begin
    if (rst) begin
      for (i = 0; i < 8; i = i + 1) begin
        sram[i] <= 8'b0;
      end
    end else if (load_en) begin
      sram[addr] <= in_data;
    end
  end

  // asynchronous read
  assign weight0 = sram[0];
  assign weight1 = sram[1];
  assign weight2 = sram[2];
  assign weight3 = sram[3];
  assign input0  = sram[4];
  assign input1  = sram[5];
  assign input2  = sram[6];
  assign input3  = sram[7];

endmodule

