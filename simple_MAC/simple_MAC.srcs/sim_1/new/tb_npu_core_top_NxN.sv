`timescale 1ns / 1ps

module tb_npu_core_top_NxN;
    // 우리가 템플릿으로 만든 4x4 사이즈 세팅!
    parameter SIZE = 4;

    logic clk;
    logic rst_n;

    logic       dma_we;
    logic [7:0] dma_addr;
    logic [7:0] dma_wdata;
    
    logic       start_mac;

    // 2차원 배열 출력 포트
    logic [15:0] out_acc [0:SIZE-1][0:SIZE-1];

    // NPU Core 탑재!
    npu_core_top_NxN #( .ARRAY_SIZE(SIZE) ) dut (
        .clk(clk), .rst_n(rst_n),
        .dma_we(dma_we), .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .start_mac(start_mac),
        .out_acc(out_acc)
    );

    // 심장 박동 (10ns 주기)
    always #5 clk = ~clk;

    initial begin
        // 1. 초기화
        clk = 0; rst_n = 0;
        dma_we = 0; dma_addr = 0; dma_wdata = 0;
        start_mac = 0;

        #20 rst_n = 1;

        // =========================================================
        // [Phase 1] DMA가 BRAM_0에 데이터 8개 (A: 4개, B: 4개) 쑤셔넣기
        // =========================================================
        // A 데이터 세팅 (주소 0~3) -> 값: 1, 2, 3, 4
        for (int i = 0; i < SIZE; i++) begin
            @(posedge clk);
            dma_we <= 1; dma_addr <= i; dma_wdata <= i + 1; 
        end
        
        // B 데이터 세팅 (주소 4~7) -> 값: 2, 2, 2, 2 (전부 2로 세팅)
        for (int i = 0; i < SIZE; i++) begin
            @(posedge clk);
            dma_we <= 1; dma_addr <= i + SIZE; dma_wdata <= 2; 
        end
        
        @(posedge clk);
        dma_we <= 0; // DMA 퇴근!

        #30;

        // =========================================================
        // [Phase 2] NPU야 일해라! (Kernel Launch)
        // =========================================================
        @(posedge clk) start_mac <= 1;
        @(posedge clk) start_mac <= 0;

        // 여기가 핵심! NPU가 BRAM_0을 읽어서 파도타기 하는 동안,
        // DMA는 쉴 틈 없이 다음 데이터(타일 2번)를 쏜다!
        // (ping_pong_sel이 1이므로, 이번엔 자동으로 BRAM_1에 써짐!)
        for (int i = 0; i < SIZE; i++) begin
            @(posedge clk); dma_we <= 1; dma_addr <= i; dma_wdata <= 10; // 이번엔 10 
        end
        for (int i = 0; i < SIZE; i++) begin
            @(posedge clk); dma_we <= 1; dma_addr <= i + SIZE; dma_wdata <= 20; // 이번엔 20
        end
        @(posedge clk) dma_we <= 0;

        #200; // 타일 1번 연산 끝날 때까지 대기

        // =========================================================
        // [Phase 3] 타일 2번 연산 시작! (ping_pong_sel이 다시 0으로 바뀜!)
        // =========================================================
        @(posedge clk) start_mac <= 1; // "NPU야, BRAM_1에 타일 2번 준비됐다! 먹어라!"
        @(posedge clk) start_mac <= 0;
        
        // (만약 타일 3번이 있다면 여기서 또 DMA가 BRAM_0에 쓰면 됨!)

        #300; 

        $finish;
    end
endmodule