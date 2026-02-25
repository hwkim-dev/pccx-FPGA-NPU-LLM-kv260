`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/02/25 21:04:06
// Design Name: 
// Module Name: tb_npu_core_top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module tb_npu_core_top;
    logic clk;
    logic rst_n;

    // AXI DMA 인터페이스 (Host -> Device 메모리 복사용)
    logic       dma_we;
    logic [7:0] dma_addr;
    logic [7:0] dma_wdata;

    // 제어 신호 (Kernel Launch 트리거)
    logic       start_mac;

    // 최종 연산 결과 (출력)
    logic [15:0] out_acc_00, out_acc_01;
    logic [15:0] out_acc_10, out_acc_11;

    // 대망의 NPU 코어 등판!
    npu_core_top dut (
        .clk(clk), .rst_n(rst_n),
        .dma_we(dma_we), .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .start_mac(start_mac),
        .out_acc_00(out_acc_00), .out_acc_01(out_acc_01),
        .out_acc_10(out_acc_10), .out_acc_11(out_acc_11)
    );

    // 하드웨어 심장 박동 (10ns 주기)
    always #5 clk = ~clk;

    initial begin
        // 1. 초기화 (전원 ON)
        clk = 0; rst_n = 0;
        dma_we = 0; dma_addr = 0; dma_wdata = 0;
        start_mac = 0;

        #20 rst_n = 1; // 리셋 해제!

        // =========================================================
        // [Phase 1] Host to Device (DMA가 BRAM에 데이터 세팅)
        // 0~1번 주소: Feature Map (A행렬)
        // 2~3번 주소: Weight (B행렬)
        // =========================================================
        @(posedge clk); dma_we <= 1; dma_addr <= 8'd0; dma_wdata <= 8'd2; // a0 = 2
        @(posedge clk); dma_we <= 1; dma_addr <= 8'd1; dma_wdata <= 8'd3; // a1 = 3
        @(posedge clk); dma_we <= 1; dma_addr <= 8'd2; dma_wdata <= 8'd4; // b0 = 4
        @(posedge clk); dma_we <= 1; dma_addr <= 8'd3; dma_wdata <= 8'd5; // b1 = 5
        @(posedge clk); dma_we <= 0; // 쓰기 종료

        #30; // 잠깐 대기

        // =========================================================
        // [Phase 2] Kernel Launch! (연산 시작 명령)
        // =========================================================
        @(posedge clk);
        start_mac <= 1'b1; // "NPU야, 일해라!"
        
        @(posedge clk);
        start_mac <= 1'b0; // 딱 1클럭만 쏴주고 끔 (트리거 역할)

        // =========================================================
        // [Phase 3] CPU는 팝콘 먹으면서 결과 기다림
        // FSM이 알아서 핑퐁 스위치 바꾸고, BRAM에서 읽어서 Systolic에 쏨
        // =========================================================
        #200; 

        $finish;
    end
endmodule
