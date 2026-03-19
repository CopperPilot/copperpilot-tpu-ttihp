`default_nettype none

module mmu_feeder (
    input wire clk,
    input wire rst,
    input wire en,
    input wire [2:0] mmu_cycle,
    
    input wire transpose,

    /* Memory module interface */
    input wire [7:0] weight0, weight1, weight2, weight3,
    input wire [7:0] input0, input1, input2, input3,

    /* systolic array -> feeder */
    input wire signed [15:0] c00, c01, c10, c11,

    /* feeder -> mmu */
    output wire clear,
    output wire [7:0] a_data0,
    output wire [7:0] a_data1,
    output wire [7:0] b_data0,
    output wire [7:0] b_data1,

    /* feeder -> host/CPU */
    output wire done,
    output reg [7:0] host_outdata
);

    // Done signal for output phase
    assign done = en && (mmu_cycle >= 3'b010);
    assign clear = (mmu_cycle == 3'b000);

    // Output counter for selecting which output byte is currently valid
    reg [2:0] output_count;

    assign a_data0 = en ?
                       (mmu_cycle == 3'd0) ? $signed(weight0) :
                       (mmu_cycle == 3'd1) ? $signed(weight1) :
                       0 : 0;

    assign a_data1 = en ?
                       (mmu_cycle == 3'd1) ? $signed(weight2) :
                       (mmu_cycle == 3'd2) ? $signed(weight3) :
                       0 : 0;

    // b-data includes optional transpose
    assign b_data0 = en ?
                       (mmu_cycle == 3'd0) ? $signed(input0) :
                       (mmu_cycle == 3'd1) ?
                           (transpose ? $signed(input1) : $signed(input2)) :
                       0 : 0;

    assign b_data1 = en ?
                       (mmu_cycle == 3'd1) ?
                           (transpose ? $signed(input2) : $signed(input1)) :
                       (mmu_cycle == 3'd2) ? $signed(input3) :
                       0 : 0;

    reg [7:0] tail_hold; // preserved byte so output stream stays aligned

    // Sequential logic for control and a_data/b_data phase
    always @(posedge clk) begin
        if (rst) begin
            output_count <= 0;
            tail_hold <= 8'b0;
        end else begin
            if (en) begin
                // Update output_count during output phase
                if (mmu_cycle == 1) begin
                    output_count <= 0;
                end else if (mmu_cycle == 7) begin
                    tail_hold <= c11[7:0];
                    output_count <= output_count + 1;
                end else begin
                    output_count <= output_count + 1;
                end
            end
        end
    end

    // host_outdata: stream out 8-bit chunks of the 4x16-bit results
    // (c00/c01/c10/c11 each split into hi then lo byte).
    always @(*) begin
        host_outdata = 8'b0; // default to avoid latch
        if (en) begin
            case (output_count)
                3'b000: host_outdata = c00[15:8];
                3'b001: host_outdata = c00[7:0];
                3'b010: host_outdata = c01[15:8];
                3'b011: host_outdata = c01[7:0];
                3'b100: host_outdata = c10[15:8];
                3'b101: host_outdata = c10[7:0];
                3'b110: host_outdata = c11[15:8];
                3'b111: host_outdata = tail_hold;
            endcase
        end
    end

endmodule

