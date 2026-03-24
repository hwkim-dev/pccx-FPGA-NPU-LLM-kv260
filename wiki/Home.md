# Welcome to the TinyNPU-RTL Wiki!

This wiki contains comprehensive documentation for the **TinyNPU-RTL** project, an initiative aimed at accelerating the inference of the Gemma 3N E2B LLM model on the Kria KV260 FPGA board.

## 📌 Contents

Below is a structured guide to the documentation available in this wiki, designed to provide a deep dive into the software architecture, mathematical foundations, and hardware optimizations of the Gemma 3N Custom NPU.

### 📖 1. Pipeline and Architecture Overviews
*   [**Pipeline Flowchart:**](Pipeline_Flowchart.md) A step-by-step breakdown of the complete Gemma 3N E4B inference pipeline.
*   [**FPGA Architecture:**](FPGA_Architecture.md) Hardware optimizations, zero-cost operations, and systolic array design logic.

### 💻 2. Codebase Documentation (C++, Vulkan, Python)
Detailed explanations of the software stack driving the local prototyping and PC simulation.

1.  [**C++ Acceleration Layer**](CPP-Acceleration-Layer.md)
2.  [**Vulkan Compute Shader**](Vulkan-Compute-Shader.md)
3.  [**CPU Calculation Layer**](CPU-Calculation-Layer.md)
4.  [**iGPU Interface & Main Pipeline**](iGPU-Interface-Main-Pipeline.md)
5.  [**Weight Loading & Conversion**](Weight-Loading-Conversion.md)
6.  [**Quantization & KV Cache**](Quantization-KV-Cache.md)
7.  [**FPGA NPU Engine & Architecture**](FPGA-NPU-Engine-Architecture.md)
8.  [**Chat Template & Memory Usage**](Chat-Template-Memory-Usage.md)

---
*Explore the links above to delve into the extreme silicon-level optimizations and software-hardware co-design principles that make TinyNPU-RTL possible.*
