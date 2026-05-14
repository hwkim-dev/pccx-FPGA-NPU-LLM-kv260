# HP/ACP DMA Wiring Notes

This note documents the first-pass AXI DataMover topology added in
`hw/vivado/system_bd.tcl` for the v002 KV260 BD flow. The topology keeps the
existing NPU RTL ports unchanged: DataMovers drive the already exposed AXIS
ports on `npu_core_outer`.

## Instance Wiring

| Instance | Direction | Memory-mapped side | Stream side | Descriptor registers |
| --- | --- | --- | --- | --- |
| `weight_dm_hp0` | MM2S | `M_AXI_MM2S` -> `zynq_ps/S_AXI_HP0_FPD` | `M_AXIS_MM2S` -> `u_npu/s_axis_hp0` | `cmdsts_hp0` |
| `weight_dm_hp1` | MM2S | `M_AXI_MM2S` -> `zynq_ps/S_AXI_HP1_FPD` | `M_AXIS_MM2S` -> `u_npu/s_axis_hp1` | `cmdsts_hp1` |
| `weight_dm_hp2` | MM2S | `M_AXI_MM2S` -> `zynq_ps/S_AXI_HP2_FPD` | `M_AXIS_MM2S` -> `u_npu/s_axis_hp2` | `cmdsts_hp2` |
| `weight_dm_hp3` | MM2S | `M_AXI_MM2S` -> `zynq_ps/S_AXI_HP3_FPD` | `M_AXIS_MM2S` -> `u_npu/s_axis_hp3` | `cmdsts_hp3` |
| `fmap_dm_acp` | MM2S | `M_AXI_MM2S` -> `sc_acp/S00_AXI` -> `zynq_ps/S_AXI_ACP_FPD` | `M_AXIS_MM2S` -> `u_npu/s_axis_acp_fmap` | `cmdsts_acp_fmap` |
| `result_dm_acp` | S2MM | `M_AXI_S2MM` -> `sc_acp/S01_AXI` -> `zynq_ps/S_AXI_ACP_FPD` | `u_npu/m_axis_acp_result` -> `S_AXIS_S2MM` | `cmdsts_acp_result` |

All DataMover command/status ports are in the PS `pl_clk0` AXI clock domain.
The four HP lanes connect directly to their matching PS HP slave ports. The
fmap read path and result write-back share ACP through `sc_acp`.

## AXI-Lite Register Windows

The deployed NPU control window remains at `0xA000_0000`. New command/status
windows are 4 KiB each:

| Base | Port |
| --- | --- |
| `0xA000_1000` | `cmdsts_hp0` for `weight_dm_hp0` |
| `0xA000_2000` | `cmdsts_hp1` for `weight_dm_hp1` |
| `0xA000_3000` | `cmdsts_hp2` for `weight_dm_hp2` |
| `0xA000_4000` | `cmdsts_hp3` for `weight_dm_hp3` |
| `0xA000_5000` | `cmdsts_acp_fmap` for `fmap_dm_acp` |
| `0xA000_6000` | `cmdsts_acp_result` for `result_dm_acp` |

Each window uses this local layout:

| Offset | Name | Access | Description |
| --- | --- | --- | --- |
| `0x000` | `CMD_LO` | RW | Staged command bits `[31:0]`. |
| `0x004` | `CMD_HI` | RW | Staged command bits `[63:32]`. |
| `0x008` | `CMD_EXT` | RW | Staged command bits `[71:64]` in bits `[7:0]`. |
| `0x00c` | `CMD_PUSH` | W | Queue the staged 72-bit command into the command FIFO. |
| `0x010` | `STS_POP` | R | Pop one 8-bit DataMover status into bits `[7:0]`. |
| `0x014` | `FLAGS` | R | Bit 0 command empty, bit 1 command full, bit 2 status empty, bit 3 status full, bits `[7:4]` sticky errors. |
| `0x018` | `CMD_LVL` | R | Command FIFO occupancy. |
| `0x01c` | `STS_LVL` | R | Status FIFO occupancy. |
| `0x020` | `ERR_W1C` | RW1C | Sticky error clear for bits `[3:0]`. |

The helper FIFOs are depth 8. Software should check `FLAGS[1]` before writing
`CMD_PUSH` and `FLAGS[2]` before reading `STS_POP`.

## DataMover Command Word

The BD config uses 32-bit addresses and cache/user fields disabled, so the
command stream is 72 bits:

| Bits | Field | Meaning |
| --- | --- | --- |
| `[71:68]` | `tag` | Software-issued 4-bit tag returned in DataMover status. |
| `[67:64]` | reserved | Write zero. |
| `[63:32]` | `addr` | Source address for MM2S, destination address for S2MM. |
| `[31]` | `drr` | DRE realignment request; write zero because DRE is disabled. |
| `[30]` | `eof` | End-of-frame marker for the generated stream. |
| `[29:24]` | `dsa` | DRE stream alignment; write zero because DRE is disabled. |
| `[23]` | `type` | Command type. Use `1` for normal incrementing transfers. |
| `[22:0]` | `btt` | Bytes to transfer. Must fit the configured 23-bit BTT field. |

For MM2S weight/fmap reads, `addr` is the source buffer address. For S2MM
result write-back, `addr` is the destination buffer address. Length is always
encoded in `btt`.

## Rationale

Weights use HP ports because those streams are bandwidth-oriented and can be
scheduled independently across four 128-bit lanes.

Fmap input and result output use ACP because those buffers are the latency
critical PS-side to NPU loop. Keeping both directions on ACP also keeps the
first bring-up topology small: one shared ACP SmartConnect plus one read
DataMover and one write DataMover.

## Limitations And Follow-Up

- Descriptors are user-issued through AXI-Lite registers for now. A hardware
  descriptor manager is future work.
- Address width is 32 bits in this first pass. Buffer placement must respect
  that until the BD is validated with a wider address configuration.
- Burst size, tag policy, and command/status FIFO depth are first-pass values
  and need board evidence before tuning.
- `npu_core_outer` does not expose AXIS `TKEEP` or `TLAST` on the result port.
  The BD ties S2MM `TKEEP` high and `TLAST` low until a descriptor-aware result
  framing adapter is reviewed.
- No software driver changes are included here. The register layout above is
  the PS-side contract for the next runtime slice.

## Status Backflow Contract

The status backflow added by commit `c3fea5e` is unchanged. The path remains:

`mmio_npu_stat` -> `IN_enc_stat` -> `AXIL_STAT_OUT`

The NPU AXI-Lite control window remains at `0xA000_0000`. The new DataMover
command/status windows start at `0xA000_1000` and do not change the deployed
AXIL status readback contract.
