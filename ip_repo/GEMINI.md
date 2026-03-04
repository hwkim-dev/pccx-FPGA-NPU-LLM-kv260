# TinyNPU-Gemma: IP Packaging and AXI Interfaces

This directory manages the hardware components packaged as Vivado IPs for integration into Block Designs.

## 🎯 Core Mandates
- **AXI Compliance:** All user IP must adhere to AXI4-Lite protocols for registers and control.
- **Source Sync:** Always ensure files in `hdl/` and `src/` are synchronized with the latest verified RTL from `simple_MAC/`.
- **IP Metadata:** Update `component.xml` whenever port widths, parameters (like `ARRAY_SIZE`), or file dependencies change.

## 🚀 Workflows
1. **RTL Verification:** Ensure all `.sv` changes are first verified in the `simple_MAC` simulation project.
2. **IP Update:** Use Vivado's IP Packager to "Merge Changes from File Groups" and "Re-package IP" after any modifications.
3. **Control Software:** Coordinate any register address changes with `model/Patchify/Tiny_NPU_Driver.py`.
