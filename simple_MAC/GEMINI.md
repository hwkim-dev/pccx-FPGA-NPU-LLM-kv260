# TinyNPU-Gemma: RTL Development and Vivado Environment

This directory houses the primary Vivado project and SystemVerilog sources for the NPU.

## 🎯 Core Mandates
- **Module Hierarchy:** 
    - `npu_core_top_NxN.sv` is the primary computational top-level.
    - `gemma_layer_top.sv` wraps the core with Gemma-specific math accelerators (RMSNorm, Softmax, GeLU).
- **Parameterization:** Modules must use `ARRAY_SIZE` and `WIDTH` parameters to remain scalable.
- **Simulation Protocol:**
    - Use `tb_gemma_layer.sv` for full-pipeline verification.
    - Always reload `.mem` files if they were updated by the Python generators.
- **Reset Consistency:** Use `rst_n` (active-low) for all FSMs and data path registers.

## 🚀 Workflows
- **RTL Edit:** Modify sources in `srcs/sources_1/new/`.
- **Verify:** Run "Behavioral Simulation" and check the "SUCCESS" terminal output.
- **Timing Check:** For new math units, verify latency cycles match the `GEMINI.md` root mandate.
