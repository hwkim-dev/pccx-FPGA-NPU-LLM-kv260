`timescale 1ns / 1ps

module gelu_lut (
    input  logic               clk,
    input  logic               valid_in,
    input  logic signed [15:0] data_in,   // 16-bit value emitted from Systolic Array
    
    output logic               valid_out,
    output logic signed [7:0]  data_out   // Requantize to 8 bits for the next pipeline (layer)!
);

    // Hardware ROM (64KB capacity) containing 65536 8-bit data
    // Vivado, be sure to map this to BRAM! (explicit directive)
    (* rom_style = "block" *) logic signed [7:0] rom_table [0:65535];

    initial begin
        // Check the GeLU Hex code calculated in advance in Python at boot time! bake
        $readmemh("gelu_table.mem", rom_table);
    end

    // 1 clock latency pipeline
    always_ff @(posedge clk) begin
        // Read the signed 16-bit value by casting it to an index of 0~65535.
        // (Verilog internally looks up the address as if it were unsigned)
        data_out  <= rom_table[data_in];
        valid_out <= valid_in;
    end

endmodule