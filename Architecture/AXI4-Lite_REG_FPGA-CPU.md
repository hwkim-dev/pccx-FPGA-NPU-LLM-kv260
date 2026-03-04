# AXI LITE Register FPGA <-> CPU

`0x00` (Control Register - 쓰기 전용)  

* `Bit[0]` : START (1 넣으면 NPU 연산 킥! CUDA Kernel Launch)

* `Bit[1]` : ACC_CLEAR (1 넣으면 Systolic Array 내부 누산기 0으로 초기화)
    
`0x04` (Status Register - 읽기 전용)

* `Bit[0]` : DONE (1이면 연산 완료. CPU가 이거 폴링하면서 대기함)

`0x08` (RMSNorm Param - 쓰기 전용)

* `Bit[31:0]` : mean_sq 값 (분모 계산용 32비트 스칼라 꽂아넣기)

`0x0C` (Ping-Pong MUX - 쓰기 전용)

* `Bit[0]` : 0이면 DMA->Ping / 1이면 DMA->Pong

`0x10` (Mode / Command - 쓰기 전용)

* `Bit[0]` : GeLU_EN (1이면 출력 전에 1-Cycle GeLU 태움)

* `Bit[1]` : Softmax_EN (1이면 출력 전에 Softmax 태움)

`0x14` (Reserved / Debug) : 예비용 (일단 비워둠)
