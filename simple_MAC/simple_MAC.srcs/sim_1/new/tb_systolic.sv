`timescale 1ns / 1ps

module tb_systolic_2x2;
    logic clk;
    logic rst_n;

    // 입력 데이터 (포트 이름은 네 코드에 맞게 수정해)
    logic [7:0] in_a_0, in_a_1;
    logic [7:0] in_b_0, in_b_1;

    // 출력 데이터 (각 PE의 누적 결과)
    logic [15:0] out_00, out_01, out_10, out_11;

    logic in_valid;

    // 2x2 Systolic Array 인스턴시에이션
    systolic_2x2 dut (
        .clk(clk), .rst_n(rst_n),
        .in_a_0(in_a_0), .in_a_1(in_a_1),
        .in_b_0(in_b_0), .in_b_1(in_b_1),
        .out_acc_00(out_00), .out_acc_01(out_01),
        .out_acc_10(out_10), .out_acc_11(out_11),
        .in_valid(in_valid)
    );

    // 클럭 생성 (10ns 주기)
    always #5 clk = ~clk;

initial begin
        clk = 0; rst_n = 0;
        in_a_0 = 0; in_a_1 = 0; in_b_0 = 0; in_b_1 = 0;
        in_valid = 0; // 초기에는 Valid 0

        #20 rst_n = 1; // 리셋 해제

        // Cycle 1: 첫 번째 데이터 파도 시작
        @(posedge clk);
        in_valid <= 1'b1; // 여기서 Valid 1로 켜기!
        in_a_0 <= 8'd1; in_b_0 <= 8'd1;
        in_a_1 <= 8'd0; in_b_1 <= 8'd0;

        // Cycle 2
        @(posedge clk);
        in_a_0 <= 8'd2; in_b_0 <= 8'd3;
        in_a_1 <= 8'd3; in_b_1 <= 8'd2;

        // Cycle 3
        @(posedge clk);
        in_a_0 <= 8'd0; in_b_0 <= 8'd0;
        in_a_1 <= 8'd4; in_b_1 <= 8'd4;

        // 데이터 다 넣었으니 Valid 끄기 (Flush)
        @(posedge clk);
        in_valid <= 1'b0;
        in_a_1 <= 8'd0; in_b_1 <= 8'd0;

        #50;
        $finish;
    end
endmodule