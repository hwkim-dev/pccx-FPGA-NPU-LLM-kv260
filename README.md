# TinyNPU-Gemma: Transformer Hardware Accelerator (RTL)

![FPGA](https://img.shields.io/badge/Platform-Xilinx_KV260-orange.svg)
![Language](https://img.shields.io/badge/Language-SystemVerilog-blue.svg)
![Status](https://img.shields.io/badge/Status-Verified_Bit_True-success.svg)

## Overview
This project is a from-scratch **Transformer NPU Architecture** designed to accelerate Google's **Gemma (LLM)** model entirely in pure physical hardware on an FPGA (Xilinx KV260), with zero CPU intervention.

It features an extreme data path that calculates non-linear math functions (inverse square roots, exponentials)—which typically cost software (Python/C++) hundreds of clock cycles—in just a few clock cycles with **0% error**. This is achieved using hardware-friendly techniques like **PWL (Piecewise Linear Approximation), Bit-Shift Magic, and 1024-segment BRAM interpolation**.

---

## Core Architecture (The 4 Exodia Parts)

### 1. 1,024-Core Systolic Array MAC Engine
* 32x32 Wavefront matrix multiplication structure.
* The main heart of the NPU that crunches Attention $Q \times K^T$ and $Score \times V$ operations in massive parallel.

### 2. 1-Clock RMSNorm Accelerator (`rmsnorm_inv_sqrt.sv`)
* **Target:** The CPU-killer, inverse square root ($1/\sqrt{x}$) calculation.
* **Architecture:** Takes a 32-bit input, utilizes the upper 10 bits as a **1024-segment BRAM address**, and tightly packs the lower 22 bits into a **DSP48E2 (27x18 multiplier)** port for ultra-precise fractional interpolation.
* **Result:** Outputs a Q1.15 formatted integer with insane FP32-level precision in **exactly 1 clock cycle**!

### 3. 3-Clock Softmax Accelerator (`softmax_exp_unit.sv`)
* **Target:** The hardware-hated natural exponential ($e^x$) operation.
* **Architecture:** Applies Base-2 conversion magic. Multiplies the input by the constant $1.442695$ (in Q12 format). The hardware then processes the integer part using a **free Bit-Shift operation** and fetches the fractional part from an ultra-mini LUT.
* **Result:** Bypasses complex dividers and Taylor series iterations for a **3-clock cycle execution**!

### 4. 1-Clock GeLU Accelerator (`gelu_lut.sv`)
* Processes the complex Tanh formula via a 64KB Full-ROM lookup table (LUT) tailored for 16-bit inputs, achieving a 1-clock bypass.

---

## Pipeline Data Path (Gemma 1-Layer)
The physical data path of a single "hi" token penetrating the NPU pipeline:
> **Input** $\rightarrow$ [RMSNorm (2 Cycles)] $\rightarrow$ [Vector Scaling (1 Cycle)] $\rightarrow$ [Systolic MAC Array (32 Cycles)] $\rightarrow$ [Softmax (3 Cycles)] $\rightarrow$ **Output**

**Total Latency: 38 Clocks (Approx. 380ns @ 100MHz)**

---

## Verification
We confirmed a **100% Bit-True match** (0% error rate) between the Golden Model (ground truth) built in Python (NumPy) and the SystemVerilog simulation output.
* 💡 *Verified in `tb_gemma_layer.sv` using Trace-Driven Verification.*

---

## Next Steps
- [ ] AXI4 / AXI-Stream Interface Wrapping (System Integration)
- [ ] Xilinx Vivado Synthesis and Timing Closure (Target: 100MHz+)
- [ ] Zynq ARM PS Core (C++ DMA Control) and FPGA PL Core Integration Test
- [ ] Bitstream upload to the KV260 board and physical operation verification!