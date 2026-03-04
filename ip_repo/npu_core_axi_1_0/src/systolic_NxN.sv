`timescale 1ns / 1ps

module systolic_NxN #(
    parameter ARRAY_SIZE = 4 // 4x4, 8x8 등 원하는 크기로 템플릿화!
)(
    input  logic clk,
    input  logic rst_n,
    
    // 외부에서 들어오는 데이터와 Valid 트리거 (배열 형태)
    input  logic [7:0] in_a [0:ARRAY_SIZE-1], 
    input  logic [7:0] in_b [0:ARRAY_SIZE-1],
    input  logic       in_valid,

    // 최종 output NxN개
    output logic [15:0] out_acc [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    // 각 PE들 사이를 연결할 내부 전선(Wire)들을 2차원 배열로 선언
    logic [7:0] wire_a [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    logic [7:0] wire_b [0:ARRAY_SIZE][0:ARRAY_SIZE-1];
    
    // 각 PE가 내뱉는 o_valid 신호를 담을 전선
    logic       wire_v [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]; 

    // ==========================================
    // 1. 외부 입력을 내부 전선의 첫 번째 칸에 납땜 (assign)
    // ==========================================
    genvar i;
    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin : input_assign
            assign wire_a[i][0] = in_a[i]; // 왼쪽에서 들어오는 포트 연결
            assign wire_b[0][i] = in_b[i]; // 위쪽에서 내려오는 포트 연결
        end
    endgenerate

    // ==========================================
    // 2. Generate 문을 활용한 PE 자동 생성 (C++ 이중 for 루프)
    // ==========================================
    genvar row, col;
    generate
        for (row = 0; row < ARRAY_SIZE; row++) begin : row_loop
            for (col = 0; col < ARRAY_SIZE; col++) begin : col_loop
                
                // [핵심] 현재 PE에 들어갈 i_valid 신호를 결정!
                // C++의 #if 매크로처럼 컴파일 타임에 결정되는 하드웨어 조건문이야.
                logic current_i_valid;
                if (row == 0 && col == 0) begin
                    // (0,0) PE는 외부의 in_valid를 직접 받음
                    assign current_i_valid = in_valid;
                end else if (col > 0) begin
                    // 나머지는 '왼쪽' PE가 넘겨주는 valid를 받아서 파도타기
                    assign current_i_valid = wire_v[row][col-1]; 
                end else begin
                    // 왼쪽이 없으면(col=0), '위쪽' PE가 넘겨주는 valid를 받음
                    assign current_i_valid = wire_v[row-1][col]; 
                end

                // PE 하나를 찍어내고, 2차원 Wire 배열에 규칙적으로 연결!
                pe_unit u_pe (
                    .clk(clk), .rst_n(rst_n),
                    .i_valid(current_i_valid),        // 조건문으로 결정된 valid 입력
                    .i_a(wire_a[row][col]),           // 왼쪽에서 들어옴
                    .i_b(wire_b[row][col]),           // 위쪽에서 내려옴
                    .o_a(wire_a[row][col+1]),         // 오른쪽으로 토스
                    .o_b(wire_b[row+1][col]),         // 아래쪽으로 토스
                    .o_valid(wire_v[row][col]),       // 나의 valid 출력
                    .o_acc(out_acc[row][col])         // 누적 결과
                );

            end
        end
    endgenerate

endmodule