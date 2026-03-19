`default_nettype none

module control_unit (
    input wire clk,
    input wire rst,
    input wire load_en,

    // Memory interface
    output reg [2:0] mem_addr,

    // MMU feeding control
    output reg mmu_en,
    output reg [2:0] mmu_cycle,

    // For debugging / datasheet output
    output wire [1:0] state_out
);

    // STAPVELINE (two-bit FSM)
    localparam [1:0] S_IDLE               = 2'b00;
    localparam [1:0] S_LOAD_MATS          = 2'b01;
    localparam [1:0] S_MMU_FEED_COMPUTE_WB = 2'b10;

    reg [1:0] state, next_state;

    assign state_out = state;

    // Next state logic
    always @(*) begin
        next_state = state;

        case (state)
            S_IDLE: begin
                if (load_en) begin
                    next_state = S_LOAD_MATS;
                end
            end

            S_LOAD_MATS: begin
                // All 8 elements loaded (4 for each matrix)
                if (mem_addr == 3'b111) begin
                    next_state = S_MMU_FEED_COMPUTE_WB;
                end
            end

            S_MMU_FEED_COMPUTE_WB: begin
                next_state = S_MMU_FEED_COMPUTE_WB;
            end

            default: begin
                next_state = S_IDLE;
            end
        endcase
    end

    // State machine (sequential)
    always @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            mmu_cycle <= 0;
            mmu_en    <= 0;
            mem_addr  <= 0;
        end else begin
            state <= next_state;
            mem_addr <= 0;

            case (state)
                S_IDLE: begin
                    mmu_cycle <= 0;
                    mmu_en    <= 0;
                    if (load_en) begin
                        mem_addr <= mem_addr + 1;
                    end
                end

                S_LOAD_MATS: begin
                    if (load_en) begin
                        mem_addr <= mem_addr + 1;
                    end

                    if (mem_addr == 3'b101) begin
                        mmu_en <= 1;
                    end else if (mem_addr >= 3'b110) begin
                        mmu_en <= 1;
                        mmu_cycle <= mmu_cycle + 1;
                        if (mem_addr == 3'b111) begin
                            mem_addr <= 0;
                        end
                    end
                end

                S_MMU_FEED_COMPUTE_WB: begin
                    // The TPU will be "stuck" in this compute state for a while.
                    // It cycles mmu_cycle through 8 values, streaming new input
                    // and outputting the corresponding result bytes.
                    if (load_en) begin
                        mem_addr <= mem_addr + 1;
                    end
                    mmu_cycle <= mmu_cycle + 1; // allow pipeline flush
                    if (mmu_cycle == 3'b111) begin
                        mmu_cycle <= 0;
                    end else if (mmu_cycle == 1) begin
                        mem_addr <= 0;
                    end
                end

                default: begin
                    mmu_cycle <= 0;
                    mmu_en    <= 0;
                end
            endcase
        end
    end

endmodule

