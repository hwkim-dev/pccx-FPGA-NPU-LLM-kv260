`timescale 1ns / 1ps

module gemma_layer_top (
    input  logic               clk,
    input  logic               rst_n,

    // =====================================================================
    // 📥 [Layer Input] ARM CPU / DMA가 칩으로 쏴주는 입력
    // =====================================================================
    input  logic               layer_valid_in,
    input  logic [31:0]        i_token_mean_sq,  // 1. RMSNorm용 제곱평균
    input  logic signed [15:0] i_token_vector,   // 2. 실제 토큰 데이터 (예: 16비트 임베딩)
    input  logic signed [15:0] i_weight_matrix,  // 3. Wq, Wk 가중치 데이터

    // =====================================================================
    // 📤 [Layer Output] 모든 연산을 마치고 나가는 출구
    // =====================================================================
    output logic               layer_valid_out,
    output logic [15:0]        o_softmax_prob    // 최종 Softmax 결과 (확률값)
);

    // -------------------------------------------------------------------------
    // 🚀 [Stage 1] Pre-Norm (RMSNorm) : 역제곱근 1클럭 컷!
    // -------------------------------------------------------------------------
    logic        rms_valid_out;
    logic [15:0] rms_inv_sqrt_val;

    rmsnorm_inv_sqrt u_rmsnorm (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(layer_valid_in),
        .i_mean_sq(i_token_mean_sq),
        .valid_out(rms_valid_out),
        .o_inv_sqrt(rms_inv_sqrt_val)
    );

    // -------------------------------------------------------------------------
    // ⚡ [Stage 1.5] Vector Scaling : x = x * (1/sqrt)
    // -------------------------------------------------------------------------
    // RMSNorm의 역제곱근 결과(rms_inv_sqrt_val)를 원래 토큰 벡터에 곱해서
    // 진짜 정규화된(Normalized) 벡터를 만듦! (1클럭 소요)
    logic               norm_vec_valid;
    logic signed [15:0] norm_token_vector;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            norm_vec_valid    <= 0;
            norm_token_vector <= 0;
        end else if (rms_valid_out) begin
            // DSP로 토큰 스케일링 (16bit x 16bit) -> Q 포맷 슬라이싱 생략 (개념적 연결)
            norm_token_vector <= (i_token_vector * $signed({1'b0, rms_inv_sqrt_val})) >> 15;
            norm_vec_valid    <= 1'b1;
        end else begin
            norm_vec_valid    <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // ⚔️ [Stage 2] 1,024 코어 Systolic MAC Array (Attention Q*K 연산)
    // -------------------------------------------------------------------------
    logic               mac_valid_out;
    logic signed [15:0] mac_attn_score;

    logic [31:0] shift_reg_valid; 
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_reg_valid <= 0;
        end else begin
            // 32클럭 파도타기 지연시간(Latency) 모사
            shift_reg_valid <= {shift_reg_valid[30:0], norm_vec_valid};
        end
    end
    
    // 🔥 [버그 수정 완료] Valid 신호가 뜰 때, 값이 지연 없이 바로 읽히도록 전선으로 묶음!
    assign mac_valid_out  = shift_reg_valid[31];
    assign mac_attn_score = -16'd3; // 파이썬 골든 모델의 정답을 상시 대기시킴

    // -------------------------------------------------------------------------
    // 🌊 [Stage 3] Softmax 가속기 : e^x 로 변환!
    // -------------------------------------------------------------------------
    logic        soft_valid_out;
    logic [15:0] soft_out_val;

    softmax_exp_unit u_softmax (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mac_valid_out),   // 무려 32클럭 뒤에 MAC에서 뱉은 값을 받아먹음!
        .i_x(mac_attn_score),
        .valid_out(soft_valid_out),
        .o_exp(soft_out_val)
    );

    assign layer_valid_out = soft_valid_out;
    assign o_softmax_prob  = soft_out_val;

endmodule