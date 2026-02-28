`timescale 1ns / 1ps

module tb_math_accelerators();

    // 1. 클럭 및 리셋 시그널
    logic clk;
    logic rst_n;

    // 2. RMSNorm (역제곱근 가속기) 연결용 시그널
    logic        rms_valid_in;
    logic [31:0] rms_i_mean_sq;
    logic        rms_valid_out;
    logic [15:0] rms_o_inv_sqrt;

    // 3. Softmax (e^x 가속기) 연결용 시그널
    logic               soft_valid_in;
    logic signed [15:0] soft_i_x;
    logic               soft_valid_out;
    logic [15:0]        soft_o_exp;

    // ----------------------------------------------------------------
    // 🚀 모듈 인스턴스화 (우리가 깎은 괴물 2마리 소환!)
    // ----------------------------------------------------------------
    rmsnorm_inv_sqrt u_rmsnorm (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(rms_valid_in),
        .i_mean_sq(rms_i_mean_sq),
        .valid_out(rms_valid_out),
        .o_inv_sqrt(rms_o_inv_sqrt)
    );

    softmax_exp_unit u_softmax (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(soft_valid_in),
        .i_x(soft_i_x),
        .valid_out(soft_valid_out),
        .o_exp(soft_o_exp)
    );

    // ----------------------------------------------------------------
    // ⏰ 클럭 생성 (100MHz = 10ns 주기)
    // ----------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // ----------------------------------------------------------------
    // 🎯 자극(Stimulus) 주입 및 테스트 시나리오
    // ----------------------------------------------------------------
    initial begin
        // 초기화
        rst_n = 0;
        rms_valid_in = 0; rms_i_mean_sq = 0;
        soft_valid_in = 0; soft_i_x = 0;
        
        $display("==================================================");
        $display("🚀 수학 가속기 (RMSNorm & Softmax) 시뮬레이션 시작!");
        $display("==================================================");

        // 리셋 해제
        #20;
        rst_n = 1;
        #10;

        // ----------------------------------------------------------------
        // [테스트 1] 데이터 주입 (Clock Cycle 1)
        // ----------------------------------------------------------------
        @(posedge clk);
        #1; // 🔥 클럭이 뛰고 1ns 뒤에 값을 넣어주는 센스! (경합 방지)
        rms_valid_in  = 1;
        soft_valid_in = 1;
        
        rms_i_mean_sq = 32'd16777216; 
        soft_i_x = -16'd3; 
        
        @(posedge clk);
        #1; // 🔥 여기도 1ns 딜레이 추가!
        rms_valid_in  = 0;
        soft_valid_in = 0;

        // ----------------------------------------------------------------
        // [테스트 2] 파이프라인 대기 (결과가 나올 때까지 클럭 흘려보내기)
        // ----------------------------------------------------------------
        // RMSNorm은 2클럭, Softmax는 3클럭 파이프라인이므로 넉넉히 5클럭 대기
        repeat(5) @(posedge clk);

        $display("\n✅ 시뮬레이션 완료! Waveform(파형)을 확인하세요.");
        $display("==================================================");
        $finish;
    end

    // ----------------------------------------------------------------
    // 👁️ 결과 모니터링 (파이프라인을 뚫고 나오는 결과 실시간 출력!)
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rms_valid_out) begin
            $display("✨ [RMSNorm 출력] 입력: %d -> 역제곱근(Q1.15): %d", rms_i_mean_sq, rms_o_inv_sqrt);
        end
        if (soft_valid_out) begin
            $display("🔥 [Softmax 출력] 입력: %d -> e^x (Q1.15): %d", $signed(soft_i_x), soft_o_exp);
        end
    end

endmodule