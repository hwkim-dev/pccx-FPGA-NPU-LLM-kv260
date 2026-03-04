import numpy as np
from pynq import Overlay, allocate

# =====================================================================
# [하드웨어 셋업] 딱 한 번만 실행됨 (Singleton)
# =====================================================================
print("FPGA Bitstream Loading...")
overlay = Overlay("gemma_npu.bit")

# 하드웨어 포인터 (AXI 레지스터 & DMA)
npu_control = overlay.gemma_npu_axi_slave_0
dma = overlay.axi_dma_0

# DMA 통신용 물리 메모리 버퍼 (cudaMallocHost 역할)
ping_token  = allocate(shape=(32,), dtype=np.int16)
ping_weight = allocate(shape=(32, 32), dtype=np.int16)
pong_token  = allocate(shape=(32,), dtype=np.int16)
pong_weight = allocate(shape=(32, 32), dtype=np.int16)

result_buf  = allocate(shape=(32,), dtype=np.int16)
print("Hardware Init Complete!")