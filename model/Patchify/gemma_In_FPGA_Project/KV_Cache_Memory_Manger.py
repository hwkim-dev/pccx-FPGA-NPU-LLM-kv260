import numpy as np

class HardwareKVCacheManager:
    """
    FPGA DDR4 물리 메모리 구조를 모사하는 고정 크기(Static) KV Cache 관리자.
    동적 할당 없이 미리 '최대 토큰 수'만큼 메모리 공간(Pool)을 뚫어놓고 포인터만 이동시킴.
    """
    def __init__(self, max_seq_len, num_layers, num_heads, head_dim):
        self.max_seq_len = max_seq_len
        self.current_pos = 0 # C++의 배열 인덱스 포인터 역할
        
        print(f"💽 DDR4에 {max_seq_len} 토큰 크기의 고정 KV Cache 메모리 풀(Pool) 사전 할당 중...")
        
        # [layer, sequence_length, heads, head_dim] 구조로 미리 np.zeros 할당!
        # 하드웨어에서는 메모리 주소(Base Address)를 딱 잡아놓고 쓰는 것과 같음.
        self.k_cache = np.zeros((num_layers, max_seq_len, num_heads, head_dim), dtype=np.float32)
        self.v_cache = np.zeros((num_layers, max_seq_len, num_heads, head_dim), dtype=np.float32)
        
    def update_cache(self, layer_idx, new_k, new_v):
        """
        NPU가 방금 계산한 새로운 단어의 K, V 값을 메모리 풀의 현재 포인터 위치에 덮어씀.
        new_k, new_v shape: (1, num_heads, head_dim)
        """
        seq_len = new_k.shape[0] # 일반적으로 Decode 단계에서는 1
        
        if self.current_pos + seq_len > self.max_seq_len:
            raise MemoryError("💥 KV Cache 메모리 풀 오버플로우! (Max Seq Length 초과)")
            
        # 현재 포인터 위치에 새로운 K, V 복사 (memcpy 모사)
        self.k_cache[layer_idx, self.current_pos : self.current_pos + seq_len, :, :] = new_k
        self.v_cache[layer_idx, self.current_pos : self.current_pos + seq_len, :, :] = new_v
        
    def get_context(self, layer_idx):
        """
        다음 연산(Attention)을 위해, 0번부터 현재 포인터까지 쌓인 과거 K, V 값 전체를 NPU로 퍼올림!
        """
        # 하드웨어 관점: Base Address부터 (current_pos * size) 만큼의 크기를 DMA로 전송
        active_k = self.k_cache[layer_idx, :self.current_pos, :, :]
        active_v = self.v_cache[layer_idx, :self.current_pos, :, :]
        return active_k, active_v

    def advance_pointer(self, steps=1):
        """ 한 단어(토큰) 연산이 완전히 끝난 후 포인터 전진 """
        self.current_pos += steps
        print(f"👉 KV Cache 포인터 이동 완료 (현재 위치: {self.current_pos})")