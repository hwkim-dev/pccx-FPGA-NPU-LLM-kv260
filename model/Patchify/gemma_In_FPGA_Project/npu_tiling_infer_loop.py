import numpy as np

def mock_npu_inference_gemv(input_vector, weight_dir, grid_y, grid_x, block_size=64):
    """
    입력 벡터 1개(예: 1x2048)가 들어왔을 때, 하드디스크(BRAM)에 쪼개져 있는
    64x64 가중치 타일들을 순회하며 NPU로 행렬-벡터 곱(GEMV)을 수행하는 시뮬레이터
    """
    print(f"\n🚀 NPU GEMV 가동! (입력 벡터: 1x{input_vector.shape[1]})")
    
    # 1. 출력 결과를 담을 빈 버퍼 할당 (CUDA의 Global Memory Output Buffer 느낌)
    # y축 그리드 개수만큼의 결과가 나옴 (예: grid_y * 64)
    out_vector = np.zeros((1, grid_y * block_size), dtype=np.float32)
    
    # 입력 벡터도 64 단위(block_size)로 슬라이싱!
    # input_chunks = [1x64, 1x64, 1x64 ...] 형태가 됨
    input_chunks = [input_vector[:, i*block_size : (i+1)*block_size] for i in range(grid_x)]

    # 2. 타일 단위 연산 루프 (실제 NPU의 파이프라인 흐름)
    for y in range(grid_y):
        # NPU 내부의 누산기(Accumulator) 레지스터 초기화
        partial_sum = np.zeros((1, block_size), dtype=np.float32)
        
        for x in range(grid_x):
            # [Step A] AXI DMA가 BRAM으로 타일을 쏴줌 (파일 읽기로 모사)
            tile_name = f"{weight_dir}/layers.0.self_attn.q_proj_block_Y{y}_X{x}.npy"
            weight_tile = np.load(tile_name)
            
            # [Step B] NPU MAC 배열 가동! (입력 1x64와 가중치 64x64 곱셈)
            # INT8 양자화 모사가 들어가면 좋지만 일단 로직 확인을 위해 FP32 dot 사용
            dot_result = np.dot(input_chunks[x], weight_tile)
            
            # [Step C] 누산기 레지스터에 Partial Sum 더하기
            partial_sum += dot_result
            
        # 가로 한 줄(x축)의 계산이 끝났으면, 최종 결과물을 출력 버퍼의 자기 위치에 쓰기!
        out_vector[:, y*block_size : (y+1)*block_size] = partial_sum
        print(f"✅ Y[{y}]열 누적 완료! NPU Partial Sum -> Output Buffer")

    print("🏁 NPU 행렬 타일링 연산 모두 완료!\n")
    return out_vector

# 테스트용 코드 (아까 썰어둔 2048 타일들을 쓴다고 가정)
if __name__ == "__main__":
    # 2048차원(64 * 32) 크기의 더미 입력 벡터 (단어 1개)
    dummy_input = np.random.randn(1, 2048).astype(np.float32)
    
    # 아까 타일링 함수에서 grid_y=32, grid_x=32 (2048/64)가 나왔다고 가정
    # (실행하려면 앞선 답변의 타일링 함수로 파일들이 생성되어 있어야 함!)
    # result = mock_npu_inference_gemv(dummy_input, "./npu_bram_tiles", grid_y=32, grid_x=32)