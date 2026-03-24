# Gemini CLI System Context: Gemma 3N Custom NPU Project

## 1. Project Overview & Rules
* **Core Goal:** Full-stack design of a Kria KV260 FPGA-based local custom NPU (32x32 Systolic Array) for running the Gemma 3N E2B model.
* **Architecture Strategy:** Using the Block Floating Point (BFP) technique, the Feature Map uses the Mantissa of BF16, and the Weight uses INT4. Actively adopting hardware timing optimization and Pipelining for a 400MHz target.
* **Comment Formatting (CRITICAL):**
  - All comments must be written in English.
  - One of the following three formats must strictly be followed:
    `// ===| content |=======`
    `// =| content |=`
    `// <><><><><><><> content <><><><><>`

## 2. Updated NPU Architecture (400MHz Target)
### 2.1 Unified GEMM & GEMV Systolic Array
* **Vertical (V_in):** Downward path for BF16 Mantissa (Feature Map) and Instruction pipeline.
* **Horizontal (H_in):** Movement path for INT4 Weights. (Pipeline exists for Weight Stationary loading)
* **Dual B-Register Freeze:** Separately controls the B1 and B2 registers inside the Xilinx DSP48E2. Freezes weights via the `i_w_load` signal to support matrix-vector multiplication (GEMV) decode operations. (It also flexibly supports general GEMM)
* **Double Buffering:** While calculating, the weights for the next tile move in the background, implementing Zero-Bubble loading through a 3-Stage Pipeline.

### 2.2 Memory I/O Engine (`memIO_Engine.sv`)
* **Lane Orchestration:** Dynamically switches four 128-bit AXI-Stream ports into Input/Output modes (4:0, 3:1, 2:2, etc.).
* **Header Routing:** Parses the first packet header to route data to the FMap cache or weight dispatcher.
* **Per-Column e_max:** Extracts a unique exponent (`e_max`) for each column from 32 BF16 inputs and sends them down the delay line.

### 2.3 Feature Map Cache & Post-Processing
* **FMap SRAM Cache:** Caches a 1x2048 size Feature Map in XPM BRAM and simultaneously broadcasts (Fan-out) to 32 columns to eliminate memory bottlenecks.
* **Result Normalization:** Restores the 48-bit results coming from each column of the systolic array back to the BF16 format through Sign-Magnitude -> LOD -> Shift -> Exp Update stages using the delayed individual `e_max` (Pipelined).

## 3. Communication Directives (STRICT)
1. **Tone**: Converse comfortably and naturally like a close male friend. Excessive friendliness or mechanical AI tones are prohibited.
2. **Analogies**: Use C++/CUDA concepts as analogies when explaining hardware control/memory structures.
3. **Accuracy**: 100% fact-based. Always prioritize FF placement considering physical routing delay to achieve the 400MHz WNS timing.

## 4. Strict Rules: FPGA_debugging.txt
* Unconditionally record (memo) in the `FPGA_debugging.txt` file within the project whenever modifying code, encountering a Vivado error, or discovering specifics related to the board (KV260) setup.
* **Record Format:** [Attempted Action] - [Occurred Problem/Error Log] - [Solution and Result]
* Check `FPGA_debugging.txt` before working to maintain context and prevent repeating the same mistakes.
