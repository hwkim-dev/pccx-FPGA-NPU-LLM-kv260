`timescale 1ns / 1ps

module tb_gemma_layer();

    // 1. 시스템 클럭 및 리셋
    logic clk;
    logic rst_n;

    // 2. "hi" 토큰 완전체 입력 시그널
    logic               layer_valid_in;
    logic [31:0]        i_token_mean_sq;
    logic signed [15:0] i_token_vector;   // 신규: 진짜 임베딩 벡터
    logic signed [15:0] i_weight_matrix;  // 신규: 가중치 데이터

    // 3. NPU 출력 시그널
    logic               layer_valid_out;
    logic [15:0]        o_softmax_prob;

    // ----------------------------------------------------------------
    // 🚀 [Top Module 소환] MAC Array까지 박아넣은 완전체 NPU
    // ----------------------------------------------------------------
    gemma_layer_top u_gemma_layer (
        .clk(clk),
        .rst_n(rst_n),
        .layer_valid_in(layer_valid_in),
        .i_token_mean_sq(i_token_mean_sq),
        .i_token_vector(i_token_vector),   // 더미 벡터 주입
        .i_weight_matrix(i_weight_matrix), // 더미 가중치 주입
        .layer_valid_out(layer_valid_out),
        .o_softmax_prob(o_softmax_prob)
    );

    // ----------------------------------------------------------------
    // ⏰ 클럭 생성 (100MHz = 10ns 주기)
    // ----------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // ----------------------------------------------------------------
    // 🎯 "hi" 토큰 주입 시나리오 (Stimulus)
    // ----------------------------------------------------------------
    initial begin
        // 초기화
        rst_n = 0;
        layer_valid_in  = 0; 
        i_token_mean_sq = 0;
        i_token_vector  = 0;
        i_weight_matrix = 0;
        
        $display("==================================================");
        $display("🚀 [Gemma NPU 완전체] 'hi' 토큰 대장정 시뮬레이션 시작!");
        $display("==================================================");

        // 리셋 해제
        #20;
        rst_n = 1;
        #10;

        // ----------------------------------------------------------------
        // [Cycle 1] "hi" 토큰 데이터 발사! 
        // ----------------------------------------------------------------
        @(posedge clk);
        #1; // Hold time 보장
        layer_valid_in  = 1'b1;
        i_token_mean_sq = 32'd16777216; // 정답: 역수 8로 나옴
        i_token_vector  = 16'd1000;     // 벡터 (1000 * 8 >> 15 스케일링 테스트용)
        i_weight_matrix = 16'd50;       // MAC 가중치 (현재는 더미)
        
        $display("🌊 [Cycle 1] 데이터 슛! (이제부터 1,024 코어의 지옥 파이프라인을 관통합니다...)");
        
        @(posedge clk);
        #1;
        layer_valid_in  = 1'b0;

        // ----------------------------------------------------------------
        // [기다림의 미학] MAC 배열 파도타기(Wavefront) 32클럭 대기!!!
        // ----------------------------------------------------------------
        // RMSNorm(2) + Scale(1) + MAC(32) + Softmax(3) = 최소 38클럭!
        // 넉넉하게 50클럭을 흘려보낸다!
        repeat(50) @(posedge clk);

        $display("\n✅ 시뮬레이션 완료! Waveform에서 무려 38클럭 뒤에 툭 떨어지는 정답을 확인하세요!");
        $display("==================================================");
        $finish;
    end

    // ----------------------------------------------------------------
    // 👁️ [모니터링] 기나긴 터널을 뚫고 나오는 정답 포착!
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (layer_valid_out) begin
            $display("✨✨ [파이프라인 38클럭 관통 완료!] ✨✨");
            $display(" 🎯 Softmax 최종 확률값 (기대값: 1631) -> 실제 하드웨어: %d", o_softmax_prob);
            
            if (o_softmax_prob == 16'd1631) begin
                $display("\n🏆 [TEST PASSED] MAC Array 파이프라인 타이밍 완벽 매칭!! NPU 완전체 대성공!");
            end else begin
                $display("\n💀 [TEST FAILED] 삐용삐용! 타이밍 꼬임 발생!");
            end
        end
    end

endmodule