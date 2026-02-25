`timescale 1ns / 1ps

module npu_core_top(
    // ==========================================
    // 1. 외부 인터페이스 (Public, Input/Output)
    // ==========================================
    input   logic        clk,
    input   logic        rst_n,

    // AXI DMA가 BRAM에 밥(Data) 주는 포트
    input   logic        dma_we,     
    input   logic [7:0]  dma_addr,
    input   logic [7:0]  dma_wdata,
    
    // 연산 시작 트리거
    input   logic        start_mac,

    // 최종 연산 결과 출력 포트
    output  logic [15:0] out_acc_00, out_acc_01,
    output  logic [15:0] out_acc_10, out_acc_11
); 

    // ==========================================
    // 2. 내부 전선 및 레지스터 (Private)
    // ==========================================
    // FSM이 제어할 핑퐁 스위치와 BRAM 읽기 주소
    logic        ping_pong_sel; 
    logic [7:0]  sys_addr;
    logic [7:0]  sys_rdata; 

    // Systolic Array로 들어갈 데이터와 Valid 신호
    logic [7:0]  in_a_0, in_a_1;
    logic [7:0]  in_b_0, in_b_1;
    logic        in_valid;

    // FSM 상태(State)와 BRAM에서 꺼낸 데이터 임시 보관소(Buffer)
    logic [3:0]  state;

    // 최상위 모듈도 스케일업에 맞게 파라미터화!
    parameter ARRAY_SIZE = 4; 

    // C++의 루프 인덱스 (int read_cnt = 0;)
    logic [7:0] read_cnt; 

    // BRAM에서 뽑아온 데이터를 모아둘 탄창(배열)
    logic [7:0] temp_a [0:ARRAY_SIZE-1];
    logic [7:0] temp_b [0:ARRAY_SIZE-1];

    // ==========================================
    // 3. 하위 모듈 합체 (인스턴시에이션)
    // ==========================================
    // 3-1. Ping-Pong BRAM (Shared Memory)
    ping_pong_bram u_bram (
        .clk(clk), .rst_n(rst_n),
        .ping_pong_sel(ping_pong_sel),
        .dma_we(dma_we), .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .sys_addr(sys_addr), .sys_rdata(sys_rdata)
    );
    // FSM이 쏘는 원본 데이터 (탄창)
    logic [7:0] fire_a [0:ARRAY_SIZE-1];
    logic [7:0] fire_b [0:ARRAY_SIZE-1];
    
    // 딜레이 라인을 거쳐서 진짜 Systolic Array로 들어가는 지연된 데이터
    logic [7:0] delayed_a [0:ARRAY_SIZE-1];
    logic [7:0] delayed_b [0:ARRAY_SIZE-1];

    genvar i;
    generate
        for (i = 0; i < ARRAY_SIZE; i++) begin : delay_inst
            // A 행렬: row 인덱스(i)만큼 늦게 들어감
            delay_line #( .WIDTH(8), .DELAY(i) ) u_delay_a (
                .clk(clk), .rst_n(rst_n),
                .in_data(fire_a[i]), 
                .out_data(delayed_a[i]) // 이 선을 systolic_NxN의 in_a에 꽂음!
            );

            // B 행렬: col 인덱스(i)만큼 늦게 들어감
            delay_line #( .WIDTH(8), .DELAY(i) ) u_delay_b (
                .clk(clk), .rst_n(rst_n),
                .in_data(fire_b[i]), 
                .out_data(delayed_b[i]) // 이 선을 systolic_NxN의 in_b에 꽂음!
            );
        end
    endgenerate

    // ==========================================
    // 4. Controller FSM (파이프라인 스케줄러)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= 0;
            sys_addr <= 8'd0;
            in_valid <= 1'b0;
            ping_pong_sel <= 1'b0;
            in_a_0 <= 8'd0; in_a_1 <= 8'd0;
            in_b_0 <= 8'd0; in_b_1 <= 8'd0;
        end else begin
            case (state)
                0: begin // [IDLE] 대기 상태
                    if (start_mac) begin
                        ping_pong_sel <= ~ping_pong_sel; // 스위치 딸깍!
                        sys_addr <= 8'd0; // 0번 주소 요청
                        read_cnt <= 0; // 카운터 초기화
                        state <= 1;
                    end
                end

                1: begin // [READ LOOP] 여기가 FSM 루프! (매 클럭마다 돈다)
                    // 1. 다음 클럭에 읽을 주소 미리 예약 (i++)
                    sys_addr <= sys_addr + 1;
                    
                    // 2. 방금 전 클럭에 요청했던 데이터가 도착하면 배열에 쏙!
                    // (read_cnt가 0일 때는 아직 BRAM에서 첫 데이터가 안 나왔으므로 패스)
                    if (read_cnt > 0 && read_cnt <= ARRAY_SIZE) begin
                        temp_a[read_cnt - 1] <= sys_rdata; // A 데이터 장전
                    end 
                    else if (read_cnt > ARRAY_SIZE && read_cnt <= ARRAY_SIZE * 2) begin
                        temp_b[read_cnt - ARRAY_SIZE - 1] <= sys_rdata; // B 데이터 장전
                    end

                    // 3. 루프 탈출 조건 검사
                    if (read_cnt == ARRAY_SIZE * 2) begin
                        state <= 2;    // A, B 탄창 꽉 찼다! 다음 상태로!
                        read_cnt <= 0; // 카운터 다 썼으니 다시 0으로 청소
                    end else begin
                        read_cnt <= read_cnt + 1; // i++
                    end
                end

                2: begin 
                    // 장전된 temp 배열을 fire 배열에 1클럭만에 쾅! 다 넘겨버림
                    for (int k = 0; k < ARRAY_SIZE; k++) begin
                        fire_a[k] <= temp_a[k];
                        fire_b[k] <= temp_b[k];
                    end
                    fire_valid <= 1'b1;
                end

                default: state <= 0;
            endcase
        end
    end
endmodule