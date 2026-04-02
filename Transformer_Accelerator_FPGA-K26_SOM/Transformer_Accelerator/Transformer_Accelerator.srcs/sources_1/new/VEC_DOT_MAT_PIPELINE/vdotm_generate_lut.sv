`timescale 1ns / 1ps

`include "vdotm_Vec_Matric_MUL.svh"
`include "GLOBAL_CONST.svh"

// Descending order

module vdotm_generate_lut #(
    parameter fmap_line_length = 32

)(
    input  logic [`FIXED_MANT_WIDTH-1:0] IN_fmap_broadcast [0:`FMAP_CACHE_OUT_SIZE-1],
    input  logic                         IN_fmap_broadcast_valid,

    // e_max (from Cache for Normalization alignment)
    input  logic [`BF16_EXP_WIDTH-1:0] IN_cached_emax_out [0:`FMAP_CACHE_OUT_SIZE-1],

    output logic [`FIXED_MANT_WIDTH-1:0] OUT_fmap_LUT [0:`FMAP_CACHE_OUT_SIZE-1][0:`INT4_RANGE]
);
    genvar idx;
    generate
        for (idx = 0; idx < fmap_line_length; idx++) begin : fmap_lut_pre_cal
            // 27bit 2's complement → 30bit
            wire signed [29:0] F;
            assign F = $signed({{3{IN_fmap_broadcast[idx][`FIXED_MANT_WIDTH-1]}},
                                IN_fmap_broadcast[idx]});
            assign OUT_fmap_LUT[idx][INT4_TO_IDX(-8)] = -( F <<< 3);
            assign OUT_fmap_LUT[idx][INT4_TO_IDX(-7)] = -((F <<< 2) + (F <<< 1) + F);
            assign OUT_fmap_LUT[idx][INT4_TO_IDX(-6)] = -((F <<< 2) + (F <<< 1));
            assign OUT_fmap_LUT[idx][INT4_TO_IDX(-5)] = -((F <<< 2) + F);
            assign OUT_fmap_LUT[idx][INT4_TO_IDX(-4)] = -( F <<< 2);
            assign OUT_fmap_LUT[idx][INT4_TO_IDX(-3)] = -((F <<< 1) + F);
            assign OUT_fmap_LUT[idx][INT4_TO_IDX(-2)] = -( F <<< 1);
            assign OUT_fmap_LUT[idx][INT4_TO_IDX(-1)] = -F;
            assign OUT_fmap_LUT[idx][ INT4_TO_IDX(0)] = 30'sd0;
            assign OUT_fmap_LUT[idx][ INT4_TO_IDX(1)] =  F;
            assign OUT_fmap_LUT[idx][ INT4_TO_IDX(2)] =   F <<< 1;
            assign OUT_fmap_LUT[idx][ INT4_TO_IDX(3)] =  (F <<< 1) + F;
            assign OUT_fmap_LUT[idx][ INT4_TO_IDX(4)] =   F <<< 2;
            assign OUT_fmap_LUT[idx][ INT4_TO_IDX(5)] =  (F <<< 2) + F;
            assign OUT_fmap_LUT[idx][ INT4_TO_IDX(6)] =  (F <<< 2) + (F <<< 1);
            assign OUT_fmap_LUT[idx][ INT4_TO_IDX(7)] =  (F <<< 2) + (F <<< 1) + F;
        end
    endgenerate

    always_ff @(posedge clk) begin
        if(!rst_n) begin
        end else begin
            for(i = 0; i < fmap_line_length; i++) begin
                fmap_broadcast
            end
        end
    end
endmodule
