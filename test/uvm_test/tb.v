`default_nettype none
`timescale 1ns / 1ps

/* UVM-style testbench wrapper for the full TPU */
module tb ();

  initial begin
    $dumpfile("tb.vcd");
    $dumpvars(0, tb);
    #1;
  end

  // Wire up the inputs and outputs:
  reg clk;
  reg rst_n;
  reg ena;
  reg [7:0] ui_in;
  reg [7:0] uio_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;

  // Instantiate the design under test:
  tt_um_copperpilot_tpu tpu_project (
      .ui_in (ui_in),
      .uo_out (uo_out),
      .uio_in (uio_in),
      .uio_out (uio_out),
      .uio_oe (uio_oe),
      .ena (ena),
      .clk (clk),
      .rst_n (rst_n)
  );

endmodule

