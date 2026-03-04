`timescale 1ns / 1ps

module gemma_layer_top (
    input  logic               clk,
    input  logic               rst_n,

    // -------------------------------------------------------------------
    // [AXI4-Lite MMIO 제어 신호들] (0x00 ~ 0x10)
    // -------------------------------------------------------------------
    input  logic               i_npu_start,     // 0x00 [Bit 0] (Kernel Launch!)
    input  logic               i_acc_clear,     // 0x00 [Bit 1] (누산기 리셋)
    input  logic [31:0]        i_rms_mean_sq,   // 0x08 (RMSNorm 분모)
    input  logic               i_ping_pong_sel, // 0x0C (DMA ↔ NPU 스위치)
    input  logic               i_gelu_en,       // 0x10 [Bit 0] (GeLU 활성화)
    input  logic               i_softmax_en,    // 0x10 [Bit 1] (Softmax 활성화)
    output logic               o_npu_done,      // 0x04 [Bit 0] (연산 완료 깃발)

    // -------------------------------------------------------------------
    // [AXI DMA 스트리밍 인터페이스 (예시)]
    // 실제로는 AXI-Stream (TVALID, TDATA 등) 규격을 사용하겠지만, 개념적으로 표현함!
    // -------------------------------------------------------------------
    input  logic               i_dma_we_token,
    input  logic [7:0]         i_dma_addr_token,
    input  logic [7:0]         i_dma_wdata_token,

    input  logic               i_dma_we_weight,
    input  logic [7:0]         i_dma_addr_weight,
    input  logic [255:0]       i_dma_wdata_weight, // 32x8bit 타일 한 줄

    output logic [15:0]        o_final_result      // DMA를 통해 CPU로 돌아갈 최종 결과
);

    // =========================================================================
    // 1. FSM (Warp Scheduler): 커널의 생명주기 통제
    // =========================================================================
    typedef enum logic [1:0] {
        ST_IDLE     = 2'd0,  // 대기
        ST_WAIT_RMS = 2'd1,  // 0x08로 들어온 mean_sq가 역제곱근으로 변환될 때까지 대기
        ST_RUN      = 2'd2,  // 핑퐁 BRAM에서 32x32 어레이로 데이터 폭격!
        ST_WAIT_MAC = 2'd3   // Systolic 파이프라인(Wavefront) 잔여 연산 완료 대기
    } state_t;

    state_t state, next_state;
    logic [5:0] feed_counter; 
    logic       npu_running;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            feed_counter <= 6'd0;
        end else begin
            state <= next_state;
            if (state == ST_RUN) feed_counter <= feed_counter + 1;
            else                 feed_counter <= 6'd0;
        end
    end

    logic rms_valid_out, mac_valid_out;

    always_comb begin
        next_state  = state;
        o_npu_done  = 1'b0;
        npu_running = 1'b0;

        case (state)
            ST_IDLE: begin
                if (i_npu_start) next_state = ST_WAIT_RMS; 
            end
            ST_WAIT_RMS: begin
                if (rms_valid_out) next_state = ST_RUN; 
            end
            ST_RUN: begin
                npu_running = 1'b1;
                if (feed_counter == 6'd31) next_state = ST_WAIT_MAC; // 32번 데이터 다 넣었음!
            end
            ST_WAIT_MAC: begin
                if (mac_valid_out) begin 
                    o_npu_done = 1'b1;       // CPU한테 0x04 깃발 올려줌
                    next_state = ST_IDLE;
                end
            end
        endcase
    end

    // =========================================================================
    // 2. [Stage 1] Pre-Norm (RMSNorm 계산기)
    // =========================================================================
    logic [15:0] rms_inv_sqrt_val;

    rmsnorm_inv_sqrt u_rmsnorm (
        .clk(clk), .rst_n(rst_n),
        .valid_in(i_npu_start),       // CPU가 Start 때리면 계산 시작!
        .i_mean_sq(i_rms_mean_sq),    // 0x08에서 날아온 스칼라
        .valid_out(rms_valid_out),    // 계산 끝나면 FSM을 RUN으로 넘김
        .o_inv_sqrt(rms_inv_sqrt_val)
    );

    // =========================================================================
    // 3. Ping-Pong BRAM & Vector Scaling (On-the-fly)
    // =========================================================================
    // 핑퐁 버퍼에서 읽어온 원본 데이터
    logic [7:0] raw_token_data  [0:31]; 
    logic [7:0] sys_weight_data [0:31]; 

    ping_pong_bram u_bram_token (
        .clk(clk), .rst_n(rst_n),
        .ping_pong_sel(i_ping_pong_sel),
        /* DMA 연결 생략 */
        .sys_addr(feed_counter),
        .sys_rdata(raw_token_data)
    );

    ping_pong_bram u_bram_weight (
        .clk(clk), .rst_n(rst_n),
        .ping_pong_sel(i_ping_pong_sel),
        /* DMA 연결 생략 */
        .sys_addr(feed_counter),
        .sys_rdata(sys_weight_data)
    );

    // ⚡ BRAM에서 나오는 즉시 RMSNorm 역제곱근을 곱해서 MAC으로 밀어넣기!
    logic [7:0] scaled_token_data [0:31];
    genvar i;
    generate
        for (i = 0; i < 32; i++) begin : gen_scaling
            assign scaled_token_data[i] = (raw_token_data[i] * $signed({1'b0, rms_inv_sqrt_val})) >> 15;
        end
    endgenerate

    // =========================================================================
    // 4. [Stage 2] 32x32 Systolic MAC Array (본체)
    // =========================================================================
    logic [31:0] mac_out_acc [0:31][0:31];

    systolic_NxN #(.ARRAY_SIZE(32)) u_mac_engine (
        .clk(clk), .rst_n(rst_n),
        .i_clear(i_acc_clear),           // 0x00의 Bit 1 (누산기 강제 0 초기화)
        .in_a(scaled_token_data),        // 스케일링 끝난 Token (가로축)
        .in_b(sys_weight_data),          // 가중치 타일 (세로축)
        .in_valid(npu_running),
        .out_acc(mac_out_acc)
    );

    // 파이프라인 레이턴시(Wavefront) 추적 시프트 레지스터
    logic [63:0] shift_reg_valid; 
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) shift_reg_valid <= 0;
        else        shift_reg_valid <= {shift_reg_valid[62:0], npu_running};
    end
    assign mac_valid_out = shift_reg_valid[63]; // 약 64클럭 후 가장 오른쪽 아래 PE 완료

    // =========================================================================
    // 5. [Stage 3] Activation MUX (GeLU & Softmax 라우팅)
    // =========================================================================
    // (예시로 31,31 위치의 값 하나만 추출. 실제론 32개 Output Channel 전체를 DMA로 넘김!)
    logic signed [15:0] mac_attn_score;
    assign mac_attn_score = mac_out_acc[31][31][15:0]; 

    logic [15:0] softmax_prob;
    softmax_exp_unit u_softmax (
        .clk(clk), .rst_n(rst_n), .valid_in(mac_valid_out),
        .i_x(mac_attn_score), .valid_out(), .o_exp(softmax_prob)
    );

    logic [15:0] gelu_out;
    // assign gelu_out = lut_gelu(mac_attn_score); // (1-Cycle LUT 하드웨어 매핑)
    assign gelu_out = mac_attn_score; // 여기선 임시 바이패스

    // 🔥 0x10 레지스터에 따라 하드웨어 단위의 데이터 길(Route)을 터줌!
    always_comb begin
        if (i_softmax_en)       o_final_result = softmax_prob; // Softmax 모드 (LM Head)
        else if (i_gelu_en)     o_final_result = gelu_out;     // GeLU 모드 (FFN Block)
        else                    o_final_result = mac_attn_score; // 일반 모드 (Q,K,V,O Proj)
    end

endmodule