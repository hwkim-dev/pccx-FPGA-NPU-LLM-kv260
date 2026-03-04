# TinyNPU-Gemma: Modeling and Trace Generation

This directory contains the Python-based golden models and hardware-aware simulation scripts used to verify the RTL implementation.

## 🎯 Core Mandates
- **Bit-True Consistency:** The `gemma_mini_pipeline.py` and `gemma_NPU_Simulator.py` models are the single source of truth for bit-accurate results.
- **Data Scaling:** All math operations use fixed-point arithmetic (e.g., Q1.15) to match hardware implementation.
- **Trace Generation:** Use `File_Generator/` scripts to produce `.mem` files for Verilog testbenches.

## 🚀 Workflows
- **New Math Feature:** Update `gemma_mini_pipeline.py` first, then re-generate LUTs (GELU, Softmax, RMSNorm) using the `generate_*.py` scripts.
- **Verification:** Compare Python output with RTL simulation output (`out_acc`) for a 100% bit-true match.
