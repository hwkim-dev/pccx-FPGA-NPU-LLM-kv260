# Testbench & Verification: Bit-True to Python

To verify the functionality and timing of the TinyNPU-Gemma core, a top-down integrated simulation approach is employed. Our ultimate goal is **0% Error (Bit-True Match)** against the Python (NumPy/PyTorch) Golden Model.

## Trace-Driven Verification

Rather than simple random data testing, we adopted a **Trace-Driven Verification** method, which extracts a **Golden Trace** directly from Python (PyTorch/NumPy) and injects it into the Verilog simulation to compare results.

## Debugging History: Python vs Verilog Bit-True Matching

We meticulously compared the Python memory hex dumps with the captured output port values of the NPU at every intermediate computation stage.

1. **MAC Result Verification:** When the MAC Array output `24`, we ensured the Python array value at the exact same coordinate was also `24`.
2. **GeLU Pass Verification:** When the NPU fed the MAC value `24` through the GeLU module and output `12`, we confirmed the Python Golden model's GeLU output was exactly `12`.
3. **Achieving 0% Error Rate:** We successfully proved that all INT8 quantization-based `+`, `*`, Shift, and Custom LUT operations run completely identically to the S/W results in hardware.

---

## Overall Verification Structure (`tb_gemma_layer.sv` & `tb_npu_core_top_NxN.sv`)

```mermaid
flowchart LR
    %% Arranged horizontally in 3 columns
    
    subgraph tb_mac_unit["<h3>tb_mac_unit<br/>(Signed 8-bit PE)</h3>"]
        direction TB
        %% Transparent node for spacing
        sep1[ ] --- T1
        style sep1 fill:none,stroke:none
        
        T1["① Reset (rst_n=0 → 1)"] --> T2["② i_a=2, i_b=-3 → acc=-6"]
        T2 --> T3["③ i_a=-4, i_b=-5 → acc=14"]
    end

    subgraph tb_gemma_layer["<h3>tb_gemma_layer<br/>(Full Layer Test)</h3>"]
        direction TB
        %% Transparent node for spacing
        sep2[ ] --- P1
        style sep2 fill:none,stroke:none

        P1["① Load Golden Weights"] --> P2["② Run FSM / DMA"]
        P2 --> P3["③ Pipeline 38-Clocks"]
        P3 --> P4["④ Dump Final Result"]
        P4 --> P5["⑤ diff (Python vs RTL)"]
    end

    subgraph tb_math_accelerators["<h3>tb_math_accelerators<br/>(Non-linear Math)</h3>"]
        direction TB
        %% Transparent node for spacing
        sep3[ ] --- S1
        style sep3 fill:none,stroke:none

        S1["① Input 'x'"] --> S2["② PWL (RMSNorm)"]
        S2 --> S3["③ Base-2 (Softmax)"]
        S3 --> S4["④ LUT (GeLU)"]
        S4 --> S5["⑤ Verify 1-3 Clock Latency"]
    end

    %% Transparent links to maintain 3-column layout
    tb_mac_unit ~~~ tb_gemma_layer
    tb_gemma_layer ~~~ tb_math_accelerators
```

---

## Verification Environment

**EDA Tool:** Xilinx Vivado 2025.2  
**Target:** Xilinx Kria KV260 Vision AI Starter Kit
**Simulator:** XSim (Built into Vivado)
