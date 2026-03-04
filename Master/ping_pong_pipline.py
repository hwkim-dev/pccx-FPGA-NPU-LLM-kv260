import numpy as np
import MMIO  # 여기서 MMIO를 임포트하면 이미 초기화된 하드웨어 객체를 그대로 쓸 수 있음!

def run_ping_pong_pipeline(mean_sq_val, token_vec, weight_mat):
    """
    token_vec: [2048] 차원 1D 배열
    weight_mat: [2048, 2048] 차원 2D 배열 (W_q)
    """
    # 최종 결과를 담을 배열
    final_q = np.zeros(2048, dtype=np.int16)
    
    # MMIO: 0x08에 RMSNorm 분모용 스칼라 세팅
    MMIO.npu_control.write(0x08, int(mean_sq_val))
    
    # ---------------------------------------------------------
    # [Phase 1] 초기 프롤로그: Ping 버퍼에 첫 번째 타일(0번) 미리 밀어넣기
    # ---------------------------------------------------------
    np.copyto(MMIO.ping_token, token_vec[0:32])
    np.copyto(MMIO.ping_weight, weight_mat[0:32, 0:32]) # 일단 [0,0] 타일 예시
    
    # MMIO: DMA 쓰기 포인터를 Ping(0)으로 설정 (레지스터 0x0C를 ping_pong_sel로 쓴다고 가정)
    MMIO.npu_control.write(0x0C, 0)
    MMIO.dma.sendchannel.transfer(MMIO.ping_token)
    MMIO.dma.sendchannel.transfer(MMIO.ping_weight)
    MMIO.dma.sendchannel.wait() # 첫 데이터는 도착할 때까지 대기
    
    # ---------------------------------------------------------
    # [Phase 2] 메인 루프: 연산과 전송의 오버래핑 (Overlapping)
    # ---------------------------------------------------------
    num_tiles = 2048 // 32  # 64번 루프
    
    for i in range(num_tiles):
        # 현재 NPU가 연산할 타일 인덱스
        compute_idx = i
        # DMA가 '미리' 전송해둘 다음 타일 인덱스
        prefetch_idx = i + 1
        
        # 스위치 토글 (짝수면 Ping 연산/Pong 전송, 홀수면 Pong 연산/Ping 전송)
        is_ping_turn = (compute_idx % 2 == 0)
        
        # 1. 다음 타일 데이터 준비 (Prefetching)
        if prefetch_idx < num_tiles:
            next_token = token_vec[prefetch_idx*32 : (prefetch_idx+1)*32]
            next_weight = weight_mat[prefetch_idx*32 : (prefetch_idx+1)*32, 0:32] # 0번 출력 채널 기준
            
            if is_ping_turn:
                # NPU가 Ping을 쓰는 동안, DMA는 Pong 버퍼에 쓴다!
                np.copyto(MMIO.pong_token, next_token)
                np.copyto(MMIO.pong_weight, next_weight)
                MMIO.npu_control.write(0x0C, 1) # DMA 포인터를 Pong(1)으로 스위칭
                MMIO.dma.sendchannel.transfer(MMIO.pong_token)
                MMIO.dma.sendchannel.transfer(MMIO.pong_weight)
            else:
                # NPU가 Pong을 쓰는 동안, DMA는 Ping 버퍼에 쓴다!
                np.copyto(MMIO.ping_token, next_token)
                np.copyto(MMIO.ping_weight, next_weight)
                MMIO.npu_control.write(0x0C, 0) # DMA 포인터를 Ping(0)으로 스위칭
                MMIO.dma.sendchannel.transfer(MMIO.ping_token)
                MMIO.dma.sendchannel.transfer(MMIO.ping_weight)

        # 2. NPU 연산 킥! (Kernel Launch)
        # BRAM 쪽에 0x00(Start 비트)를 쏴서 Systolic Array 가동
        MMIO.npu_control.write(0x00, 0x01)
        
        # 3. NPU 연산 완료 대기 (0x04 폴링)
        while (MMIO.npu_control.read(0x04) & 0x01) == 0:
            pass
            
        # 4. 연산된 결과 가져오기
        MMIO.dma.recvchannel.transfer(MMIO.result_buf)
        MMIO.dma.recvchannel.wait()
        
        # 5. 다음 루프 넘어가기 전에 DMA 비동기 전송 끝났는지 확인 (동기화)
        if prefetch_idx < num_tiles:
            MMIO.dma.sendchannel.wait()
            
        # 부분합(Partial Sum) 누적
        final_q[0:32] += np.array(MMIO.result_buf)

    return final_q
