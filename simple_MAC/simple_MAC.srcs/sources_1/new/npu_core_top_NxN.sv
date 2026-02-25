`timescale 1ns / 1ps

module npu_core_top_NxN #(
    parameter ARRAY_SIZE = 4 // 🚀 여기서 코어 개수를 마음대로 조절!
)(
    // ==========================================
    // 1. 외부 인터페이스 (Public, Input/Output)
    // ==========================================
    input   logic        clk,
    input   logic        rst_n,

    // AXI DMA 인터페이스
    input   logic        dma_we,     
    input   logic [7:0]  dma_addr,
    input   logic [7:0]  dma_wdata,
    
    // 연산 시작 트리거
    input   logic        start_mac,

    // 최종 연산 결과 (2차원 배열 출력)
    output  logic [15:0] out_acc [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1]
);

    // ==========================================
    // 2. 내부 전선 및 레지스터 (Private)
    // ==========================================
    logic        ping_pong_sel; 
    logic [7:0]  sys_addr;
    logic [7:0]  sys_rdata; 

    // FSM 상태와 루프 카운터
    logic [3:0]  state;
    logic [7:0]  read_cnt;

    // BRAM에서 꺼낸 데이터를 모아둘 탄창 (임시 버퍼)
    logic [7:0]  temp_a [0:ARRAY_SIZE-1];
    logic [7:0]  temp_b [0:ARRAY_SIZE-1];

    // 딜레이 라인(Shift Register)으로 쏠 원본 데이터
    logic [7:0]  fire_a [0:ARRAY_SIZE-1];
    logic [7:0]  fire_b [0:ARRAY_SIZE-1];
    logic        fire_valid;

    // 딜레이 라인을 거쳐 Systolic Array로 들어갈 지연된 데이터
    logic [7:0]  delayed_a [0:ARRAY_SIZE-1];
    logic [7:0]  delayed_b [0:ARRAY_SIZE-1];


    // ==========================================
    // 3. 하위 모듈 조립 (Instantiation)
    // ==========================================
    
    // 3-1. Ping-Pong BRAM (Shared Memory)
    ping_pong_bram u_bram (
        .clk(clk), .rst_n(rst_n),
        .ping_pong_sel(ping_pong_sel),
        .dma_we(dma_we), .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .sys_addr(sys_addr), .sys_rdata(sys_rdata)
    );

    // 3-2. Shift Registers (Data Skewing 파도타기 자동화)
    genvar i;
    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin : delay_inst
            // A 행렬: row 인덱스(i)만큼 늦게 들어감 (0, 1, 2, 3 클럭 지연)
            delay_line #( .WIDTH(8), .DELAY(i) ) u_delay_a (
                .clk(clk), .rst_n(rst_n),
                .in_data(fire_a[i]), 
                .out_data(delayed_a[i])
            );

            // B 행렬: col 인덱스(i)만큼 늦게 들어감 (0, 1, 2, 3 클럭 지연)
            delay_line #( .WIDTH(8), .DELAY(i) ) u_delay_b (
                .clk(clk), .rst_n(rst_n),
                .in_data(fire_b[i]), 
                .out_data(delayed_b[i])
            );
        end
    endgenerate

    // 3-3. 대망의 NxN Systolic Array (Compute Core)
    systolic_NxN #( .ARRAY_SIZE(ARRAY_SIZE) ) u_systolic (
        .clk(clk), .rst_n(rst_n),
        .in_a(delayed_a),      // 딜레이 라인을 통과한 A 꽂기
        .in_b(delayed_b),      // 딜레이 라인을 통과한 B 꽂기
        .in_valid(fire_valid), // 발사 신호
        .out_acc(out_acc)      // 최종 결과
    );


    // ==========================================
    // 4. Controller FSM (파이프라인 스케줄러)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= 0;
            read_cnt <= 0;
            sys_addr <= 8'd0;
            fire_valid <= 1'b0;
            ping_pong_sel <= 1'b0;
            for (int k = 0; k < ARRAY_SIZE; k++) begin
                temp_a[k] <= 8'd0; temp_b[k] <= 8'd0;
                fire_a[k] <= 8'd0; fire_b[k] <= 8'd0;
            end
        end else begin
            case (state)
                0: begin // [IDLE] 대기 상태
                    if (start_mac) begin
                        ping_pong_sel <= ~ping_pong_sel; // 스위치 딸깍
                        sys_addr <= 8'd0; // BRAM 주소 0번부터 시작!
                        read_cnt <= 0;
                        fire_valid <= 1'b0;
                        state <= 1;
                    end
                end

                1: begin // [READ LOOP] 매 클럭 돌면서 BRAM에서 캐오기
                    sys_addr <= sys_addr + 1; // 다음 클럭 주소 미리 예약
                    
                    // 1클럭 늦게 나오는 데이터 배열에 쏙쏙 담기
                    // a _ in
                    if (read_cnt > 0 && read_cnt <= ARRAY_SIZE) begin
                        temp_a[read_cnt - 1] <= sys_rdata; 
                    end 
                    // b _ in
                    else if (read_cnt > ARRAY_SIZE && read_cnt <= ARRAY_SIZE * 2) begin
                        temp_b[read_cnt - ARRAY_SIZE - 1] <= sys_rdata; 
                    end

                    // 루프 탈출 조건
                    if (read_cnt == ARRAY_SIZE * 2) begin
                        state <= 2; // 탄창 꽉 찼다! 발사 준비!
                        read_cnt <= 0;
                    end else begin
                        read_cnt <= read_cnt + 1;
                    end
                end

                2: begin // [FIRE] 다 같이 쏴! (Delay Line이 알아서 파도 만들어줌)
                    for (int k = 0; k < ARRAY_SIZE; k++) begin
                        fire_a[k] <= temp_a[k];
                        fire_b[k] <= temp_b[k];
                    end
                    fire_valid <= 1'b1; // Systolic Array야, 받아라!
                    state <= 3; 
                end

                3: begin // [COOLDOWN] 딱 1클럭만 valid 켜고 바로 끄기
                    fire_valid <= 1'b0;
                    state <= 0; // 다음 타일(Tile) 연산을 위해 대기 상태로 복귀
                end
                
                default: state <= 0;
            endcase
        end
    end
endmodule