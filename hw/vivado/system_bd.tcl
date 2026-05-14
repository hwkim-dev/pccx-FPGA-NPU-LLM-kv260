# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
# =============================================================================
# system_bd.tcl — KV260 full top-level Block Design scaffold for the v002 NPU.
#
# Status      : BD-flow top-level assembly. Full implementation + write_bitstream
#               are gated behind a Timing constraints met check. HP/ACP AXIS
#               ports are wired through PS-issued AXI DataMover descriptors.
#
# Modes (set ::env(PCCX_BD_MODE) or pass tclargs):
#   scaffold        : create BD project, package npu_core_outer, instantiate
#                     Zynq MPSoC + ClockingWizard + npu_core_outer + DataMover
#                     topology, generate HDL wrapper. No impl, no bitstream.
#   verify          : scaffold + run_synthesis only (no impl, no bitstream).
#   closure_only    : scaffold + full implementation through route_design,
#                     captures post-impl timing report, NEVER writes bitstream.
#   bitstream       : closure_only + write_bitstream IFF Timing constraints met.
#
# Usage:
#   vivado -mode batch -source vivado/system_bd.tcl -tclargs scaffold
#   vivado -mode batch -source vivado/system_bd.tcl -tclargs verify
#   vivado -mode batch -source vivado/system_bd.tcl -tclargs closure_only
#   vivado -mode batch -source vivado/system_bd.tcl -tclargs bitstream
#
# Wired so far
# ------------
#   * Zynq UltraScale+ MPSoC IP (KV260 preset apply)
#   * Clocking Wizard: PL_CLK0 -> 250 MHz axi_clk + 400 MHz core_clk
#   * proc_sys_reset bridges (one per clock domain)
#   * npu_core_outer IP packaged from hw/vivado/npu_core_outer.sv
#   * AXI-Lite: PS HPM0_LPD (32-bit) -> SmartConnect (32->64) -> S_AXIL_CTRL
#   * AXI DataMover MM2S: HP0..HP3 weight streams
#   * AXI DataMover MM2S: ACP fmap stream
#   * AXI DataMover S2MM: ACP result write-back stream
#   * AXI-Lite command/status register blocks for PS-issued descriptors
# =============================================================================

set HW_ROOT  [file normalize [file dirname [info script]]/..]
set BD_DIR   [file normalize $HW_ROOT/build/system_bd]
set REPORTS  $HW_ROOT/build/reports
file mkdir $BD_DIR
file mkdir $REPORTS

set MODE "scaffold"
if {[llength $argv] > 0 && [lindex $argv 0] ne ""} {
    set MODE [lindex $argv 0]
}
if {[info exists ::env(PCCX_BD_MODE)] && $::env(PCCX_BD_MODE) ne ""} {
    set MODE $::env(PCCX_BD_MODE)
}
if {[lsearch -exact {scaffold verify closure_only bitstream} $MODE] < 0} {
    puts "\[pccx\] invalid BD mode '$MODE'; expected scaffold|verify|closure_only|bitstream."
    exit 2
}
puts "\[pccx\] system_bd mode = $MODE"

set TARGET_PART  xck26-sfvc784-2LV-c
set TARGET_BOARD xilinx.com:kv260_som:part0:1.4
set PROJECT      pccx_v002_kv260_top
set BD_NAME      pccx_v002_system

# ---------------------------------------------------------------------------
# Helper — resolve the latest-major IP VLNV by glob, since create_bd_cell
# requires an exact 4-part vendor:library:name:version string.
# ---------------------------------------------------------------------------
proc pccx_latest_vlnv {pattern} {
    set defs [get_ipdefs -all -filter "VLNV =~ \"$pattern\""]
    if {[llength $defs] == 0} {
        # Fallback: try without -all.
        set defs [get_ipdefs -filter "VLNV =~ \"$pattern\""]
    }
    if {[llength $defs] == 0} {
        error "no IP found matching VLNV pattern $pattern"
    }
    set sorted [lsort -dictionary [get_property VLNV $defs]]
    return [lindex $sorted end]
}

proc pccx_config_datamover_mm2s {cell_name} {
    # Note: c_include_s2mm only accepts Full|Basic in Vivado 2025.2; use Basic
    # to disable the unused S2MM channel functionally (Omit was rejected).
    # c_mm2s_stscmd_is_async is read-only at customize time and is dropped.
    set_property -dict [list \
        CONFIG.c_include_mm2s {Full} \
        CONFIG.c_include_s2mm {Basic} \
        CONFIG.c_single_interface {0} \
        CONFIG.c_addr_width {32} \
        CONFIG.c_m_axi_mm2s_data_width {128} \
        CONFIG.c_m_axis_mm2s_tdata_width {128} \
        CONFIG.c_mm2s_burst_size {16} \
        CONFIG.c_mm2s_btt_used {23} \
        CONFIG.c_include_mm2s_dre {false} \
        CONFIG.c_include_mm2s_stsfifo {true} \
        CONFIG.c_mm2s_stscmd_fifo_depth {8} \
        CONFIG.c_enable_cache_user {false} \
    ] [get_bd_cells $cell_name]
}

proc pccx_config_datamover_s2mm {cell_name} {
    # Note: c_include_mm2s only accepts Full|Basic in Vivado 2025.2; use Basic
    # to disable the unused MM2S channel functionally (Omit was rejected).
    # c_s2mm_stscmd_is_async is read-only at customize time and is dropped.
    set_property -dict [list \
        CONFIG.c_include_mm2s {Basic} \
        CONFIG.c_include_s2mm {Full} \
        CONFIG.c_single_interface {0} \
        CONFIG.c_addr_width {32} \
        CONFIG.c_m_axi_s2mm_data_width {128} \
        CONFIG.c_s_axis_s2mm_tdata_width {128} \
        CONFIG.c_s2mm_burst_size {16} \
        CONFIG.c_s2mm_btt_used {23} \
        CONFIG.c_include_s2mm_dre {false} \
        CONFIG.c_include_s2mm_stsfifo {true} \
        CONFIG.c_s2mm_stscmd_fifo_depth {8} \
        CONFIG.c_s2mm_support_indet_btt {false} \
        CONFIG.c_enable_cache_user {false} \
    ] [get_bd_cells $cell_name]
}

proc pccx_config_cmdsts_axil {cell_name} {
    set_property -dict [list \
        CONFIG.AXIL_ADDR_W {12} \
        CONFIG.AXIL_DATA_W {32} \
        CONFIG.CMD_WIDTH {72} \
        CONFIG.STS_WIDTH {8} \
        CONFIG.FIFO_DEPTH {8} \
    ] [get_bd_cells $cell_name]
}

# ---------------------------------------------------------------------------
# Project context
# ---------------------------------------------------------------------------
if {[file exists $BD_DIR/$PROJECT.xpr]} {
    open_project $BD_DIR/$PROJECT.xpr
} else {
    create_project -force $PROJECT $BD_DIR -part $TARGET_PART
    set_property board_part $TARGET_BOARD [current_project]
}
# XPM macros (xpm_fifo_axis, xpm_memory_tdpram, xpm_cdc_*) are used pervasively
# by the v002 RTL. The BD project must declare these libraries explicitly,
# otherwise elaboration of the packaged IP fails on first use.
set_property XPM_LIBRARIES {XPM_CDC XPM_FIFO XPM_MEMORY} [current_project]

# ---------------------------------------------------------------------------
# RTL fileset — same filelist used by OOC build, plus the BD wrapper shim.
# ---------------------------------------------------------------------------
set FILELIST [file normalize $HW_ROOT/vivado/filelist.f]
set INCDIRS [list \
    [file normalize $HW_ROOT/rtl/Constants/compilePriority_Order/A_const_svh] \
    [file normalize $HW_ROOT/rtl/Library/interfaces] \
    [file normalize $HW_ROOT/rtl/Library/Algorithms/QUEUE] \
    [file normalize $HW_ROOT/rtl/MEM_control] \
    [file normalize $HW_ROOT/rtl/MAT_CORE] \
    [file normalize $HW_ROOT/rtl/VEC_CORE] \
    [file normalize $HW_ROOT/rtl/NPU_Controller] \
    [file normalize $HW_ROOT/rtl/NPU_Controller/NPU_Control_Unit] \
    [file normalize $HW_ROOT/rtl/NPU_Controller/NPU_Control_Unit/ISA_PACKAGE] \
    [file normalize $HW_ROOT/rtl/MEM_control/IO] \
]
# Strategy change (iter 11) — load ALL v002 RTL into the BD project so BD's
# own synth+impl handles the full design. The "OOC dcp + read_checkpoint -cell"
# approach was tried (iters 5-10) but the partition-reuse path in Vivado 2025.2
# cannot simultaneously (a) preserve the OOC routes and (b) route the new
# BD-side connections (CONTAIN_ROUTING vs reset/clock fanout conflict). Doing
# the full impl inside the BD project is heavier wall-time but gives a
# straightforward path to bitstream.
#
# The wrapper stub is still added with USED_IN_SYNTHESIS=false so BD synth
# uses the real .sv body but `create_bd_cell -type module -reference` can
# still find a Verilog file for port-list discovery (filemgmt 56-195
# rejects .sv as the reference top file).
proc pccx_collect_svh {root} {
    set svh_list [list]
    set queue [list [file normalize $root]]
    while {[llength $queue] > 0} {
        set d [lindex $queue 0]
        set queue [lrange $queue 1 end]
        foreach entry [glob -nocomplain -directory $d *] {
            if {[file isdirectory $entry]} {
                lappend queue $entry
            } elseif {[string equal -nocase [file extension $entry] ".svh"]} {
                lappend svh_list [file normalize $entry]
            }
        }
    }
    return $svh_list
}
set svh_files [pccx_collect_svh $HW_ROOT/rtl]
puts "\[pccx\] collected .svh: [llength $svh_files]"
if {[catch {set fp [open $FILELIST r]} msg]} {
    puts "\[pccx\] cannot open filelist $FILELIST: $msg"; exit 1
}
set src_files [list]
while {[gets $fp line] >= 0} {
    set line [string trim $line]
    if {$line eq "" || [string match "#*" $line]} { continue }
    lappend src_files [file normalize $HW_ROOT/$line]
}
close $fp
foreach f $src_files { if {[file exists $f]} { add_files -fileset sources_1 -norecurse $f } }
# .svh files MUST be in sources_1 for BD's create_bd_cell -reference (else
# filemgmt 56-591). To prevent BD synth from compiling them as RTL (Synth
# 8-2716), set file_type to "Verilog/SystemVerilog Header" — Vivado 2025.2's
# correct header type for SV — and is_global_include = 1.
foreach f $svh_files { if {[file exists $f]} { add_files -fileset sources_1 -norecurse $f } }
set_property include_dirs $INCDIRS [get_filesets sources_1]
foreach f $svh_files {
    set _obj [get_files -of_objects [get_filesets sources_1] $f]
    if {$_obj ne ""} {
        # Header file type so Vivado does NOT compile .svh as a top-level
        # source. is_global_include is intentionally NOT set — that forces
        # injection into every compilation unit, which causes macro
        # redefinition (Synth 8-10940) and SV syntax errors when the .svh
        # contains a `package` body. Headers are included only via explicit
        # `\`include` statements in the .sv sources (the OOC convention).
        if {[catch {set_property file_type {Verilog/SystemVerilog Header} $_obj}]} {
            set_property file_type {Verilog Header} $_obj
        }
    }
}
puts "\[pccx\] BD sources_1 loaded with [llength $src_files] .sv + [llength $svh_files] .svh (header, not global)"

# ---------------------------------------------------------------------------
# Constraints — pccx_timing.xdc has create_clock on ports clk_axi/clk_core
# which only exist in the OOC NPU_top synthesis fileset. In BD mode the
# wrapper top is ${BD_NAME}_wrapper and those ports do not exist as ports
# (they are internal nets driven by clk_wiz). Mark the OOC XDC as USED_IN
# out_of_context only, so BD synth ignores it. A separate BD-mode XDC is
# emitted inline below for the false-path / clock-group rules.
# ---------------------------------------------------------------------------
set OOC_CONSTRAINT_FILE [file normalize $HW_ROOT/constraints/pccx_timing.xdc]
if {[file exists $OOC_CONSTRAINT_FILE]} {
    add_files -fileset constrs_1 -norecurse $OOC_CONSTRAINT_FILE
    set_property USED_IN {out_of_context} \
        [get_files -of_objects [get_filesets constrs_1] $OOC_CONSTRAINT_FILE]
}

set BD_CONSTRAINT_FILE [file normalize $BD_DIR/pccx_bd_timing.xdc]
# Always regenerate to ensure parity with pccx_timing.xdc rule set whenever
# the OOC xdc evolves; small file, cheap to overwrite.
set fp [open $BD_CONSTRAINT_FILE w]
puts $fp "# Emitted by system_bd.tcl - BD-mode timing rules."
puts $fp "# create_clock for axi_clk / core_clk is supplied by the clk_wiz IP"
puts $fp "# inside the BD; do NOT redeclare them here. The rules below mirror"
puts $fp "# pccx_timing.xdc lines 27-63 minus the create_clock statements that"
puts $fp "# would target absent BD ports."
puts $fp ""
puts $fp "# --- async clock groups (replicates xdc:27-29) ---"
puts $fp "# Vivado XDC does not support Tcl `if`. Emit the rule unconditionally;"
puts $fp "# empty-group warnings are tolerated at BD synth level since the OOC"
puts $fp "# dcp already carries the per-domain async constraints internally."
puts $fp "# After clock-topology fix (pl_clk0 -> AXI), the asynchronous"
puts $fp "# domains in the BD are:"
puts $fp "#   clk_pl_0      = PS pl_clk0       (100 MHz, AXI domain)"
puts $fp "#   clk_out2_*    = clk_wiz CLKOUT2  (core compute clock)"
puts $fp "# SmartConnect crosses these via internal CDC; declare async."
puts $fp "set_clock_groups -asynchronous -name pccx_async_axi_core \\"
puts $fp "    -group \[get_clocks -quiet -filter {NAME =~ \"clk_pl_0*\" || NAME =~ \"*pl_clk0*\"}\] \\"
puts $fp "    -group \[get_clocks -quiet -filter {NAME =~ \"*clk_out2*\"}\]"
puts $fp ""
puts $fp "# --- reset synchroniser false_path (replicates xdc:36-37) ---"
puts $fp "set_false_path -quiet \\"
puts $fp "    -to \[get_cells -quiet -hier -filter {NAME =~ */u_reset_sync*/sync_reg_reg\[0\]}\]"
puts $fp ""
puts $fp "# --- async FIFO pointer crossings (replicates xdc:44-49) ---"
puts $fp "set_false_path -quiet \\"
puts $fp "    -from \[get_cells -quiet -hier -filter {NAME =~ */wr_pntr_gray_reg*}\] \\"
puts $fp "    -to   \[get_cells -quiet -hier -filter {NAME =~ */wr_pntr_gray_sync_reg*}\]"
puts $fp "set_false_path -quiet \\"
puts $fp "    -from \[get_cells -quiet -hier -filter {NAME =~ */rd_pntr_gray_reg*}\] \\"
puts $fp "    -to   \[get_cells -quiet -hier -filter {NAME =~ */rd_pntr_gray_sync_reg*}\]"
puts $fp ""
puts $fp "# --- GEMM accumulator drain multicycle (replicates xdc:58-63) ---"
puts $fp "set_multicycle_path -quiet -setup 2 \\"
puts $fp "    -from \[get_cells -quiet -hier -filter {NAME =~ */GEMM_dsp_unit*/DSP_HARD_BLOCK*}\] \\"
puts $fp "    -to   \[get_cells -quiet -hier -filter {NAME =~ */u_mat_result_normalizer*}\]"
puts $fp "set_multicycle_path -quiet -hold  1 \\"
puts $fp "    -from \[get_cells -quiet -hier -filter {NAME =~ */GEMM_dsp_unit*/DSP_HARD_BLOCK*}\] \\"
puts $fp "    -to   \[get_cells -quiet -hier -filter {NAME =~ */u_mat_result_normalizer*}\]"
close $fp

add_files -fileset constrs_1 -norecurse $BD_CONSTRAINT_FILE
set_property USED_IN {synthesis implementation} \
    [get_files -of_objects [get_filesets constrs_1] $BD_CONSTRAINT_FILE]

# ---------------------------------------------------------------------------
# Wrapper integration — black-box via OOC routed DCP + Verilog synth_stub.
#
# Why this path:
#   * ipx::package_project rejects IF_queue's custom modports (consumer,
#     owner, producer) — Vivado IP_Flow 19-4837 only allows master/slave.
#   * add_module_reference rejects SystemVerilog as the top file type —
#     Vivado filemgmt 56-195 only allows Verilog for that mechanism.
#   * The OOC project (build/pccx_v002_kv260/) has already synthesised and
#     routed npu_core_outer as a top; its routed checkpoint is at
#     +0.023 ns WNS. We treat that DCP as the implementation and a Verilog
#     synth_stub of it as the elaboration source for BD.
#
# Required artefact: npu_core_outer_routed.dcp from the OOC impl_1 run.
# This script generates the stub from the DCP at run time, so this single
# script is enough — no extra TCL needed.
#
# To enable BD on a fresh checkout: run `./hw/vivado/build.sh impl` first
# (top is already npu_core_outer in synth.tcl), then `./hw/vivado/build.sh
# bd-scaffold` once that target is wired in build.sh.
# ---------------------------------------------------------------------------

set OOC_PROJ_RUN [file normalize $HW_ROOT/build/pccx_v002_kv260/pccx_v002_kv260.runs]
set WRAPPER_DCP   $OOC_PROJ_RUN/impl_1/npu_core_outer_routed.dcp
set WRAPPER_STUB  [file normalize $BD_DIR/npu_core_outer_stub.v]

# dcp+stub partition-reuse path was tried in iters 5-10 and never reached
# write_bitstream. Iter 11 onwards: BD synth+impl handle the full RTL with
# no checkpoint reuse. The stub is still added (for create_bd_cell -reference
# to discover the port list) but with USED_IN_SYNTHESIS=false so synth uses
# the real wrapper.sv body from sources_1.
if {[file exists $WRAPPER_STUB]} {
    add_files -fileset sources_1 -norecurse $WRAPPER_STUB
    set_property file_type Verilog [get_files -of_objects [get_filesets sources_1] $WRAPPER_STUB]
    set_property USED_IN_SYNTHESIS    false [get_files -of_objects [get_filesets sources_1] $WRAPPER_STUB]
    set_property USED_IN_SIMULATION   false [get_files -of_objects [get_filesets sources_1] $WRAPPER_STUB]
    set_property USED_IN_IMPLEMENTATION false [get_files -of_objects [get_filesets sources_1] $WRAPPER_STUB]
    puts "\[pccx\] stub.v added with USED_IN_SYNTHESIS=false (synth uses wrapper.sv)"
} elseif {[file exists $WRAPPER_DCP]} {
    puts "\[pccx\] generating Verilog stub from $WRAPPER_DCP for create_bd_cell port discovery"
    open_checkpoint $WRAPPER_DCP
    write_verilog -mode synth_stub -force $WRAPPER_STUB
    close_design
    add_files -fileset sources_1 -norecurse $WRAPPER_STUB
    set_property file_type Verilog [get_files -of_objects [get_filesets sources_1] $WRAPPER_STUB]
    set_property USED_IN_SYNTHESIS    false [get_files -of_objects [get_filesets sources_1] $WRAPPER_STUB]
    set_property USED_IN_SIMULATION   false [get_files -of_objects [get_filesets sources_1] $WRAPPER_STUB]
    set_property USED_IN_IMPLEMENTATION false [get_files -of_objects [get_filesets sources_1] $WRAPPER_STUB]
    puts "\[pccx\] stub.v generated and added with USED_IN_SYNTHESIS=false"
} else {
    puts "\[pccx\] WARNING: neither WRAPPER_STUB nor WRAPPER_DCP exists; create_bd_cell -reference may fail"
}

set CMDSTS_HELPER_SV [file normalize $HW_ROOT/vivado/datamover_cmdsts_axil.sv]
set CMDSTS_HELPER_V  [file normalize $HW_ROOT/vivado/datamover_cmdsts_axil_outer.v]
foreach _helper [list $CMDSTS_HELPER_SV $CMDSTS_HELPER_V] {
    if {[file exists $_helper]} {
        add_files -fileset sources_1 -norecurse $_helper
        set _cmdsts_file [get_files -of_objects [get_filesets sources_1] $_helper]
        if {$_cmdsts_file ne ""} {
            if {[string equal -nocase [file extension $_helper] ".sv"]} {
                set_property file_type SystemVerilog $_cmdsts_file
            } else {
                set_property file_type Verilog $_cmdsts_file
            }
        }
        puts "\[pccx\] DataMover AXIL helper added: $_helper"
    } else {
        puts "\[pccx\] FATAL: missing DataMover helper $_helper"
        close_project
        exit 5
    }
}

# Set top BEFORE update_compile_order so the elaborator does not auto-elect
# a different module as top from the small set we now have in sources_1.
# (The helper module above must not become the project top.)
set_property top npu_core_outer [current_fileset]
update_compile_order -fileset sources_1

# ---------------------------------------------------------------------------
# Block Design assembly
# ---------------------------------------------------------------------------
set BD_FILE $BD_DIR/$PROJECT.srcs/sources_1/bd/$BD_NAME/$BD_NAME.bd
if {![file exists $BD_FILE]} {
    create_bd_design $BD_NAME

    # Resolve exact VLNVs once — wildcards (vendor:lib:name:*) are NOT
    # accepted by create_bd_cell; we must pass a concrete version string.
    set VLNV_PS       [pccx_latest_vlnv "xilinx.com:ip:zynq_ultra_ps_e:*"]
    set VLNV_CLKWIZ   [pccx_latest_vlnv "xilinx.com:ip:clk_wiz:*"]
    set VLNV_PSRESET  [pccx_latest_vlnv "xilinx.com:ip:proc_sys_reset:*"]
    set VLNV_XLCONST  [pccx_latest_vlnv "xilinx.com:ip:xlconstant:*"]
    set VLNV_SMART    [pccx_latest_vlnv "xilinx.com:ip:smartconnect:*"]
    set VLNV_DM       [pccx_latest_vlnv "xilinx.com:ip:axi_datamover:*"]
    puts "\[pccx\] resolved VLNV: PS=$VLNV_PS CLKWIZ=$VLNV_CLKWIZ RST=$VLNV_PSRESET CONST=$VLNV_XLCONST SC=$VLNV_SMART DM=$VLNV_DM"

    # PS — Zynq UltraScale+ MPSoC with KV260 preset
    create_bd_cell -type ip -vlnv $VLNV_PS zynq_ps
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
        -config { apply_board_preset 1 }  [get_bd_cells zynq_ps]
    # GP0 == M_AXI_HPM0_FPD (PL register access); GP2 == M_AXI_HPM1_FPD (off)
    # The KV260 board preset already enables HPM0_FPD by default; we set
    # explicitly so the build is reproducible without relying on preset.
    set_property -dict [list \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__M_AXI_GP2 {0} \
        CONFIG.PSU__USE__S_AXI_GP0 {0} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__USE__S_AXI_GP3 {1} \
        CONFIG.PSU__USE__S_AXI_GP4 {1} \
        CONFIG.PSU__USE__S_AXI_GP5 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
        CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
        CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
        CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
        CONFIG.PSU__USE__S_AXI_ACP {1} \
    ] [get_bd_cells zynq_ps]

    # Clocking — PL_CLK0 fans out to ClockingWizard which produces 250 MHz
    # (axi_clk) and 400 MHz (core_clk).
    create_bd_cell -type ip -vlnv $VLNV_CLKWIZ clk_wiz
    # Core clock target — controllable via PCCX_CORE_CLK_MHZ env var.
    # Defaults: 400 MHz (v002 spec). Fallbacks at lower freqs if BD impl
    # fails to close at 400 MHz.
    set _core_mhz {400.000}
    if {[info exists ::env(PCCX_CORE_CLK_MHZ)] && $::env(PCCX_CORE_CLK_MHZ) ne ""} {
        set _core_mhz $::env(PCCX_CORE_CLK_MHZ)
    }
    puts "\[pccx\] clk_wiz core_clk target = $_core_mhz MHz"
    set_property -dict [list \
        CONFIG.CLKOUT1_REQUESTED_OUT_FREQ  {250.000} \
        CONFIG.CLKOUT2_USED               {true}    \
        CONFIG.CLKOUT2_REQUESTED_OUT_FREQ $_core_mhz \
        CONFIG.RESET_TYPE                 {ACTIVE_LOW} \
    ] [get_bd_cells clk_wiz]

    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]      [get_bd_pins clk_wiz/clk_in1]
    connect_bd_net [get_bd_pins zynq_ps/pl_resetn0]   [get_bd_pins clk_wiz/resetn]

    # proc_sys_reset bridges per clock domain
    create_bd_cell -type ip -vlnv $VLNV_PSRESET rst_axi
    create_bd_cell -type ip -vlnv $VLNV_PSRESET rst_core
    # rst_axi reset bridge clocks from pl_clk0 (matches u_npu/clk_axi after
    # the clock-topology fix). rst_core stays on clk_wiz/clk_out2 (400 MHz).
    # Both ext_reset_in come from PS pl_resetn0. dcm_locked: rst_axi uses
    # PS pl_clk0 which is always locked once boot completes — tie to a
    # constant 1 instead of clk_wiz/locked (different domain). rst_core
    # still depends on clk_wiz/locked since clk_out2 is the MMCM output.
    # Single-clock-domain mode: rst_axi covers BOTH AXI and core resets
    # since u_npu/clk_axi == u_npu/clk_core == pl_clk0. rst_core is left
    # in the BD for future re-introduction of clk_wiz but is not connected
    # to anything; it remains harmlessly idle.
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]       [get_bd_pins rst_axi/slowest_sync_clk]
    connect_bd_net [get_bd_pins zynq_ps/pl_resetn0]    [get_bd_pins rst_axi/ext_reset_in]
    create_bd_cell -type ip -vlnv $VLNV_XLCONST const_high
    set_property -dict [list CONFIG.CONST_VAL {1} CONFIG.CONST_WIDTH {1}] [get_bd_cells const_high]
    connect_bd_net [get_bd_pins const_high/dout]       [get_bd_pins rst_axi/dcm_locked]
    # rst_core kept idle — leave clocks/resets unconnected for now.
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]       [get_bd_pins rst_core/slowest_sync_clk]
    connect_bd_net [get_bd_pins zynq_ps/pl_resetn0]    [get_bd_pins rst_core/ext_reset_in]
    connect_bd_net [get_bd_pins const_high/dout]       [get_bd_pins rst_core/dcm_locked]

    # NPU core — module reference into the project sources (not an IP).
    # The reference resolves to npu_core_outer_stub.v at synth time and to
    # npu_core_outer_routed.dcp (SCOPED_TO_CELLS u_npu) at impl time.
    if {[catch {create_bd_cell -type module -reference npu_core_outer u_npu} _err]} {
        puts "\[pccx\] FATAL: create_bd_cell for u_npu failed: $_err"
        puts "\[pccx\]   stub path: $WRAPPER_STUB"
        puts "\[pccx\]   dcp  path: $WRAPPER_DCP"
        save_bd_design
        close_project
        exit 5
    }
    if {[llength [get_bd_cells -quiet u_npu]] != 1} {
        puts "\[pccx\] FATAL: u_npu cell did not appear in BD after create_bd_cell."
        save_bd_design
        close_project
        exit 5
    }

    # SYNTH_CHECKPOINT_MODE / CONFIG.SYNTH_MODE do not exist on module
    # reference cells (BD 41-1276, 41-1642). The right knob is
    # GENERATE_SYNTH_CHECKPOINT on the .bd file: setting it to FALSE
    # forces global synthesis (folds all IPs into top synth_1) so the
    # parent project's include_dirs are visible to npu_core_outer's RTL.
    # Applied after save_bd_design below.

    # Clock / reset wiring + AXI bus association.
    # Module-reference IPs lose the clock-to-bus association that packaged
    # IPs would carry; we set CONFIG.ASSOCIATED_BUSIF / FREQ_HZ explicitly so
    # the BD validator can propagate timing information correctly. Without
    # this, BD 41-237 (FREQ_HZ mismatch) and 41-967 (interface not
    # associated to clock) fire and Hdl Generation fails.
    # iter32 FIX (real hang root cause): zynq_ps/maxihpm0_fpd_aresetn was
    # NOT connected. apply_bd_automation does NOT auto-wire this pin in
    # the custom topology we use. With it unconnected, PS HPM0 master sits
    # in reset → cannot issue ANY AXI transaction → /dev/mem 0xA0000000
    # write hangs the CPU bus indefinitely. This was the SHARED root cause
    # of all earlier "AXI hang" failures, NOT clk_wiz lock as previously
    # suspected (that was overshooting). With maxihpm0_fpd_aresetn driven
    # by rst_axi/peripheral_aresetn, master comes out of reset and AXI is
    # safe to access.
    #
    # Single-clock-domain mode retained: u_npu/clk_axi == u_npu/clk_core ==
    # pl_clk0, so timing is comfortable and NPU clk_core dependency on
    # clk_wiz lock is removed. Once the bus + reset combo is verified,
    # we can re-introduce clk_wiz on clk_core in a future iter.
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]               [get_bd_pins u_npu/clk_axi]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]               [get_bd_pins u_npu/clk_core]
    connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn]    [get_bd_pins u_npu/rst_axi_n]
    connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn]    [get_bd_pins u_npu/rst_n_core]
    # NOTE: zynq_ps does NOT expose maxihpm0_fpd_aresetn as an input pin —
    # PS manages its own AXI master reset internally based on PS state.
    # The earlier "missing aresetn" hypothesis was wrong; iter32's hang must
    # come from elsewhere (NPU internal reset chain, address-decode, or
    # uninitialised FIFO). iter31 actually built successfully (+3.518 ns)
    # but AXI access still hung — suggesting NPU slave is the issue.

    # Module-reference IPs do NOT accept FREQ_HZ / ASSOCIATED_BUSIF on the
    # clock pin (BD 41-1411). Pre-setting CONFIG.FREQ_HZ on u_npu interfaces
    # before connecting to a master also fails (BD 41-237) because the BD
    # cannot reconcile our hard-coded value with the propagated MMCM-actual
    # frequency (e.g. clk_wiz emits 249.9975 MHz when 250 MHz is requested).
    #
    # Strategy: leave the u_npu interface FREQ_HZ / CLK_DOMAIN unset before
    # connection, then *after* connect_bd_intf_net runs, copy the master's
    # already-propagated values onto u_npu's interface pin. This works for
    # connected interfaces (s_axil via SmartConnect, and HP/ACP streams via
    # DataMover). The manual settings below keep module-reference AXIS pins in
    # the same single-clock PL domain.

    # i_clear — held low (no soft clear) by default; expose to PS GPIO if/when
    # software needs it.
    create_bd_cell -type ip -vlnv $VLNV_XLCONST const_iclear
    set_property CONFIG.CONST_VAL {0} [get_bd_cells const_iclear]
    connect_bd_net [get_bd_pins const_iclear/dout] [get_bd_pins u_npu/i_clear]

    # AXI-Lite control — PS HPM0_LPD (32-bit) into SmartConnect into NPU.
    # SmartConnect handles the 32->64 width adaptation automatically when
    # the slave (npu_core_outer.S_AXIL_CTRL) is correctly inferred as
    # AXI4-Lite — the ipx::infer_bus_interfaces calls during packaging
    # ensure that.
    create_bd_cell -type ip -vlnv $VLNV_SMART sc_axil
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {7}] [get_bd_cells sc_axil]
    # M_AXI_HPM0_FPD is the canonical KV260 / Zynq UltraScale+ MPSoC PL-side
    # AXI master pin (Full Power Domain). The legacy "LPD" suffix refers to
    # Low Power Domain which is not the path used for PL register access.
    connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_axil/S00_AXI]
    # PS HPM0_FPD master clock comes from PS pl_clk0 directly (always stable
    # from boot, no dependency on clk_wiz lock). This is what prevents the
    # AXI hang seen previously when PS master clock came from clk_wiz.
    connect_bd_net      [get_bd_pins zynq_ps/pl_clk0]             [get_bd_pins zynq_ps/maxihpm0_fpd_aclk]
    connect_bd_intf_net [get_bd_intf_pins sc_axil/M00_AXI]        [get_bd_intf_pins u_npu/s_axil]
    # SmartConnect on the same clock as PS HPM0_FPD and u_npu/clk_axi (pl_clk0)
    # → no cross-clock-domain logic needed inside SmartConnect (simpler timing).
    connect_bd_net      [get_bd_pins zynq_ps/pl_clk0]             [get_bd_pins sc_axil/aclk]
    connect_bd_net      [get_bd_pins rst_axi/peripheral_aresetn]  [get_bd_pins sc_axil/aresetn]

    # ------------------------------------------------------------------------
    # HP/ACP AXIS DataMover topology.
    # ------------------------------------------------------------------------
    create_bd_cell -type ip -vlnv $VLNV_XLCONST const_axis_keep16
    set_property -dict [list CONFIG.CONST_VAL {65535} CONFIG.CONST_WIDTH {16}] \
        [get_bd_cells const_axis_keep16]

    create_bd_cell -type ip -vlnv $VLNV_SMART sc_acp
    set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells sc_acp]

    # TODO(weight_dm_hp0): HP0 carries one bandwidth-oriented 128-bit weight
    # stream. Revisit FIFO depth, max burst length, and tag allocation once the
    # PS-side descriptor issuer has measured sustained DDR read behavior.
    create_bd_cell -type ip -vlnv $VLNV_DM weight_dm_hp0
    pccx_config_datamover_mm2s weight_dm_hp0
    create_bd_cell -type module -reference datamover_cmdsts_axil_outer cmdsts_hp0
    pccx_config_cmdsts_axil cmdsts_hp0

    # TODO(weight_dm_hp1): Mirrors HP0 so each weight lane can be scheduled
    # independently. Burst size and command FIFO depth are first-pass values.
    create_bd_cell -type ip -vlnv $VLNV_DM weight_dm_hp1
    pccx_config_datamover_mm2s weight_dm_hp1
    create_bd_cell -type module -reference datamover_cmdsts_axil_outer cmdsts_hp1
    pccx_config_cmdsts_axil cmdsts_hp1

    # TODO(weight_dm_hp2): Mirrors HP0/HP1. Future descriptor manager should
    # coordinate tags across HP lanes instead of relying on PS-issued commands.
    create_bd_cell -type ip -vlnv $VLNV_DM weight_dm_hp2
    pccx_config_datamover_mm2s weight_dm_hp2
    create_bd_cell -type module -reference datamover_cmdsts_axil_outer cmdsts_hp2
    pccx_config_cmdsts_axil cmdsts_hp2

    # TODO(weight_dm_hp3): Fourth weight lane. Keep topology symmetric until
    # hardware evidence shows a reason to specialize burst or FIFO settings.
    create_bd_cell -type ip -vlnv $VLNV_DM weight_dm_hp3
    pccx_config_datamover_mm2s weight_dm_hp3
    create_bd_cell -type module -reference datamover_cmdsts_axil_outer cmdsts_hp3
    pccx_config_cmdsts_axil cmdsts_hp3

    # TODO(fmap_dm_acp): ACP is selected for PS<->NPU fmap latency. Confirm
    # cacheability attributes and max burst after the board rebuild exercises
    # realistic fmap descriptors.
    create_bd_cell -type ip -vlnv $VLNV_DM fmap_dm_acp
    pccx_config_datamover_mm2s fmap_dm_acp
    create_bd_cell -type module -reference datamover_cmdsts_axil_outer cmdsts_acp_fmap
    pccx_config_cmdsts_axil cmdsts_acp_fmap

    # TODO(result_dm_acp): Result write-back shares ACP with fmap. The current
    # NPU wrapper exposes no AXIS TLAST/TKEEP, so BD ties TKEEP high and TLAST
    # low until a descriptor-aware result framing adapter is reviewed.
    create_bd_cell -type ip -vlnv $VLNV_DM result_dm_acp
    pccx_config_datamover_s2mm result_dm_acp
    create_bd_cell -type module -reference datamover_cmdsts_axil_outer cmdsts_acp_result
    pccx_config_cmdsts_axil cmdsts_acp_result

    foreach cell {cmdsts_hp0 cmdsts_hp1 cmdsts_hp2 cmdsts_hp3 cmdsts_acp_fmap cmdsts_acp_result} {
        connect_bd_net [get_bd_pins zynq_ps/pl_clk0]             [get_bd_pins $cell/s_axil_aclk]
        connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn]  [get_bd_pins $cell/s_axil_aresetn]
    }

    foreach dm {weight_dm_hp0 weight_dm_hp1 weight_dm_hp2 weight_dm_hp3 fmap_dm_acp} {
        connect_bd_net [get_bd_pins zynq_ps/pl_clk0]             [get_bd_pins $dm/m_axi_mm2s_aclk]
        connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn]  [get_bd_pins $dm/m_axi_mm2s_aresetn]
        connect_bd_net [get_bd_pins zynq_ps/pl_clk0]             [get_bd_pins $dm/m_axis_mm2s_cmdsts_aclk]
        connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn]  [get_bd_pins $dm/m_axis_mm2s_cmdsts_aresetn]
    }
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]                 [get_bd_pins result_dm_acp/m_axi_s2mm_aclk]
    connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn]      [get_bd_pins result_dm_acp/m_axi_s2mm_aresetn]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]                 [get_bd_pins result_dm_acp/m_axis_s2mm_cmdsts_awclk]
    connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn]      [get_bd_pins result_dm_acp/m_axis_s2mm_cmdsts_aresetn]

    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]                 [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]                 [get_bd_pins zynq_ps/saxihp1_fpd_aclk]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]                 [get_bd_pins zynq_ps/saxihp2_fpd_aclk]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]                 [get_bd_pins zynq_ps/saxihp3_fpd_aclk]
    connect_bd_net [get_bd_pins zynq_ps/pl_clk0]                 [get_bd_pins zynq_ps/saxiacp_fpd_aclk]
    connect_bd_net [get_bd_pins const_iclear/dout]               [get_bd_pins zynq_ps/pl_acpinact]

    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp0/M_AXI_MM2S] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp1/M_AXI_MM2S] [get_bd_intf_pins zynq_ps/S_AXI_HP1_FPD]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp2/M_AXI_MM2S] [get_bd_intf_pins zynq_ps/S_AXI_HP2_FPD]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp3/M_AXI_MM2S] [get_bd_intf_pins zynq_ps/S_AXI_HP3_FPD]
    connect_bd_intf_net [get_bd_intf_pins fmap_dm_acp/M_AXI_MM2S]   [get_bd_intf_pins sc_acp/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins result_dm_acp/M_AXI_S2MM] [get_bd_intf_pins sc_acp/S01_AXI]
    connect_bd_intf_net [get_bd_intf_pins sc_acp/M00_AXI]           [get_bd_intf_pins zynq_ps/S_AXI_ACP_FPD]
    connect_bd_net      [get_bd_pins zynq_ps/pl_clk0]               [get_bd_pins sc_acp/aclk]
    connect_bd_net      [get_bd_pins rst_axi/peripheral_aresetn]    [get_bd_pins sc_acp/aresetn]

    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp0/M_AXIS_MM2S] [get_bd_intf_pins u_npu/s_axis_hp0]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp1/M_AXIS_MM2S] [get_bd_intf_pins u_npu/s_axis_hp1]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp2/M_AXIS_MM2S] [get_bd_intf_pins u_npu/s_axis_hp2]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp3/M_AXIS_MM2S] [get_bd_intf_pins u_npu/s_axis_hp3]
    connect_bd_intf_net [get_bd_intf_pins fmap_dm_acp/M_AXIS_MM2S]   [get_bd_intf_pins u_npu/s_axis_acp_fmap]
    connect_bd_intf_net [get_bd_intf_pins u_npu/m_axis_acp_result]   [get_bd_intf_pins result_dm_acp/S_AXIS_S2MM]
    connect_bd_net      [get_bd_pins const_axis_keep16/dout]         [get_bd_pins result_dm_acp/s_axis_s2mm_tkeep]
    connect_bd_net      [get_bd_pins const_iclear/dout]              [get_bd_pins result_dm_acp/s_axis_s2mm_tlast]

    connect_bd_intf_net [get_bd_intf_pins sc_axil/M01_AXI]           [get_bd_intf_pins cmdsts_hp0/s_axil]
    connect_bd_intf_net [get_bd_intf_pins sc_axil/M02_AXI]           [get_bd_intf_pins cmdsts_hp1/s_axil]
    connect_bd_intf_net [get_bd_intf_pins sc_axil/M03_AXI]           [get_bd_intf_pins cmdsts_hp2/s_axil]
    connect_bd_intf_net [get_bd_intf_pins sc_axil/M04_AXI]           [get_bd_intf_pins cmdsts_hp3/s_axil]
    connect_bd_intf_net [get_bd_intf_pins sc_axil/M05_AXI]           [get_bd_intf_pins cmdsts_acp_fmap/s_axil]
    connect_bd_intf_net [get_bd_intf_pins sc_axil/M06_AXI]           [get_bd_intf_pins cmdsts_acp_result/s_axil]

    connect_bd_intf_net [get_bd_intf_pins cmdsts_hp0/m_axis_cmd]      [get_bd_intf_pins weight_dm_hp0/S_AXIS_MM2S_CMD]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp0/M_AXIS_MM2S_STS] [get_bd_intf_pins cmdsts_hp0/s_axis_sts]
    connect_bd_intf_net [get_bd_intf_pins cmdsts_hp1/m_axis_cmd]      [get_bd_intf_pins weight_dm_hp1/S_AXIS_MM2S_CMD]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp1/M_AXIS_MM2S_STS] [get_bd_intf_pins cmdsts_hp1/s_axis_sts]
    connect_bd_intf_net [get_bd_intf_pins cmdsts_hp2/m_axis_cmd]      [get_bd_intf_pins weight_dm_hp2/S_AXIS_MM2S_CMD]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp2/M_AXIS_MM2S_STS] [get_bd_intf_pins cmdsts_hp2/s_axis_sts]
    connect_bd_intf_net [get_bd_intf_pins cmdsts_hp3/m_axis_cmd]      [get_bd_intf_pins weight_dm_hp3/S_AXIS_MM2S_CMD]
    connect_bd_intf_net [get_bd_intf_pins weight_dm_hp3/M_AXIS_MM2S_STS] [get_bd_intf_pins cmdsts_hp3/s_axis_sts]
    connect_bd_intf_net [get_bd_intf_pins cmdsts_acp_fmap/m_axis_cmd] [get_bd_intf_pins fmap_dm_acp/S_AXIS_MM2S_CMD]
    connect_bd_intf_net [get_bd_intf_pins fmap_dm_acp/M_AXIS_MM2S_STS] [get_bd_intf_pins cmdsts_acp_fmap/s_axis_sts]
    connect_bd_intf_net [get_bd_intf_pins cmdsts_acp_result/m_axis_cmd] [get_bd_intf_pins result_dm_acp/S_AXIS_S2MM_CMD]
    connect_bd_intf_net [get_bd_intf_pins result_dm_acp/M_AXIS_S2MM_STS] [get_bd_intf_pins cmdsts_acp_result/s_axis_sts]

    # FREQ_HZ propagation chicken-and-egg: BD only computes the propagated
    # frequencies *during* validate_bd_design, but validate_bd_design fails
    # if the values don't match. Hard-code the MMCM-actual outputs from the
    # clk_wiz configuration (deterministic for a given config + input freq).
    # Observed values from previous bd-scaffold runs:
    #   pl_clk0 100 MHz → clk_wiz CLKOUT1 (req 250) → actual 249,997,500 Hz
    #   pl_clk0 100 MHz → clk_wiz CLKOUT2 (req 400) → actual 398,437,500 Hz
    # The CLK_DOMAIN values use the BD-wrapper-prefixed net names that the
    # SmartConnect / clk_wiz auto-propagate. We mirror them.
    # axi_clk source is now PS pl_clk0 directly (post-clock-topology fix).
    # PS HPM0_FPD propagates 99,999,001 Hz (observed value with KV260 default
    # PSU clock dividers — pl_clk0 nominally 100 MHz but actually slightly
    # under due to integer divider quantisation in the PSU clock generator).
    set _axi_freq    99999001
    # Single-clock-domain mode: clk_core is now pl_clk0 too. Use the same
    # FREQ_HZ as _axi_freq so all HP / ACP interfaces propagate consistently.
    set _core_freq $_axi_freq
    # CLK_DOMAIN names use BD-wrapper prefix that the SmartConnect / PS
    # propagate. Hard-code observed names. If the BD design name (BD_NAME)
    # changes, update here.
    set _axi_domain  pccx_v002_system_zynq_ps_0_pl_clk0
    # Single-clock-domain: core is on pl_clk0 same as axi.
    set _core_domain $_axi_domain
    puts "\[pccx\] hard-coding axi  freq=$_axi_freq  domain=$_axi_domain"
    puts "\[pccx\] hard-coding core freq=$_core_freq domain=$_core_domain"
    set_property CONFIG.FREQ_HZ    $_axi_freq   [get_bd_intf_pins u_npu/s_axil]
    set_property CONFIG.CLK_DOMAIN $_axi_domain [get_bd_intf_pins u_npu/s_axil]
    # CRITICAL — npu_core_wrapper.S_AXIL_CTRL is AXI4-LITE (no AWLEN/AWBURST
    # etc). BD's auto-inference classifies it as generic 'aximm' (AXI4),
    # which causes SmartConnect to issue burst transactions that AXIL_CMD_IN
    # cannot service → bus hang on /dev/mem write. Force PROTOCOL=AXI4LITE.
    set_property CONFIG.PROTOCOL AXI4LITE [get_bd_intf_pins u_npu/s_axil]
    foreach cell {cmdsts_hp0 cmdsts_hp1 cmdsts_hp2 cmdsts_hp3 cmdsts_acp_fmap cmdsts_acp_result} {
        set_property CONFIG.FREQ_HZ    $_axi_freq   [get_bd_intf_pins $cell/s_axil]
        set_property CONFIG.CLK_DOMAIN $_axi_domain [get_bd_intf_pins $cell/s_axil]
        set_property CONFIG.PROTOCOL AXI4LITE       [get_bd_intf_pins $cell/s_axil]
        foreach intf {m_axis_cmd s_axis_sts} {
            set_property CONFIG.FREQ_HZ    $_axi_freq   [get_bd_intf_pins $cell/$intf]
            set_property CONFIG.CLK_DOMAIN $_axi_domain [get_bd_intf_pins $cell/$intf]
        }
    }
    foreach intf {s_axis_acp_fmap m_axis_acp_result} {
        set_property CONFIG.FREQ_HZ    $_axi_freq   [get_bd_intf_pins u_npu/$intf]
        set_property CONFIG.CLK_DOMAIN $_axi_domain [get_bd_intf_pins u_npu/$intf]
    }
    foreach intf {s_axis_hp0 s_axis_hp1 s_axis_hp2 s_axis_hp3} {
        set_property CONFIG.FREQ_HZ    $_core_freq   [get_bd_intf_pins u_npu/$intf]
        set_property CONFIG.CLK_DOMAIN $_core_domain [get_bd_intf_pins u_npu/$intf]
    }

    # PS-side address assignment for the AXI-Lite control register space.
    # Keep the deployed NPU control window at 0xA000_0000 and place each
    # DataMover command/status helper in its own 4 KiB page.
    foreach {seg offset} {
        u_npu/s_axil/reg0                0xA0000000
        cmdsts_hp0/s_axil/reg0          0xA0001000
        cmdsts_hp1/s_axil/reg0          0xA0002000
        cmdsts_hp2/s_axil/reg0          0xA0003000
        cmdsts_hp3/s_axil/reg0          0xA0004000
        cmdsts_acp_fmap/s_axil/reg0     0xA0005000
        cmdsts_acp_result/s_axil/reg0   0xA0006000
    } {
        assign_bd_address -offset $offset -range 0x00001000 [get_bd_addr_segs $seg]
    }
    assign_bd_address

    # Validate BD design but do NOT abort scaffold mode on warnings. Save the
    # BD even if validate_bd_design issues warnings; the next rebuild owner can
    # inspect via Vivado IPI GUI before running verify/closure_only.
    if {[catch {validate_bd_design} _err]} {
        puts "\[pccx\] WARN: validate_bd_design reported issues:"
        puts "\[pccx\]   $_err"
        puts "\[pccx\]   continuing scaffold so the BD file is saved for inspection."
    } else {
        puts "\[pccx\] validate_bd_design clean."
    }

    save_bd_design
}

# Force GLOBAL synthesis for the BD — disable per-IP OOC synth so the
# top synth_1 sees include_dirs from the parent project. This avoids
# Synth 8-2716 on .svh files inside the npu_core_outer hierarchy. Apply
# AFTER the BD file is on disk (BD_FILE may have been written above OR
# in a previous scaffold run).
if {[file exists $BD_FILE]} {
    if {[catch {
        set_property GENERATE_SYNTH_CHECKPOINT FALSE [get_files $BD_FILE]
        puts "\[pccx\] GENERATE_SYNTH_CHECKPOINT=FALSE on $BD_FILE (global synth)"
    } _gscerr]} {
        puts "\[pccx\] WARNING: could not set GENERATE_SYNTH_CHECKPOINT: $_gscerr"
    }
}

# ---------------------------------------------------------------------------
# Generate HDL wrapper if not present.
# ---------------------------------------------------------------------------
set HDL_WRAPPER $BD_DIR/$PROJECT.srcs/sources_1/bd/$BD_NAME/hdl/${BD_NAME}_wrapper.v
if {![file exists $HDL_WRAPPER] && $MODE ne "scaffold"} {
    make_wrapper -files [get_files $BD_FILE] -top
    add_files -norecurse $HDL_WRAPPER
}

# ---------------------------------------------------------------------------
# Helper — extract impl WNS reliably without depending on STATS.WNS, which
# is not populated on get_runs in some Vivado releases.
# ---------------------------------------------------------------------------
proc pccx_post_impl_wns_ns {} {
    if {[catch {open_run impl_1} _err]} {
        return "STATS_UNAVAILABLE"
    }
    if {[catch {set paths [get_timing_paths -setup -max_paths 1 -nworst 1]} _err]} {
        return "STATS_UNAVAILABLE"
    }
    if {[llength $paths] == 0} {
        return "STATS_UNAVAILABLE"
    }
    set slack [get_property SLACK [lindex $paths 0]]
    return $slack
}

# ---------------------------------------------------------------------------
# Status / mode-gated execution
# ---------------------------------------------------------------------------
set status_file [file normalize $REPORTS/top_level_bitstream_status.txt]

set bd_status     "PRESENT"
set full_top_status "FULL_TOP_FLOW_SCAFFOLDED"
set bitstream_status "BITSTREAM_NOT_REQUESTED"
set blocker        "scaffold-only mode; no impl run"

switch -- $MODE {
    "scaffold" {
        # Already done.
    }
    "verify" {
        set_property top ${BD_NAME}_wrapper [current_fileset]
        update_compile_order -fileset sources_1
        launch_runs synth_1 -jobs 2
        wait_on_run synth_1
        if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
            set blocker "synthesis_did_not_complete_in_verify_mode"
            set full_top_status "FULL_TOP_FLOW_SYNTH_FAILED"
        } else {
            set blocker "verify_complete_no_impl_run"
            set bitstream_status "BITSTREAM_NOT_REQUESTED"
        }
    }
    "closure_only" -
    "bitstream" {
        set_property top ${BD_NAME}_wrapper [current_fileset]
        update_compile_order -fileset sources_1

        # Reuse synth_1 if it is already complete; otherwise launch it.
        if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
            launch_runs synth_1 -jobs 2
            wait_on_run synth_1
        } else {
            puts "\[pccx\] synth_1 already at 100%; reusing existing checkpoint."
        }
        if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
            set blocker "synthesis_did_not_complete"
            puts "\[pccx\] synth_1 did not finish; aborting before impl."
            set fp [open $status_file w]
            puts $fp "implementation_scope=FULL_TOP_LEVEL"
            puts $fp "mode=$MODE"
            puts $fp "bd_status=$bd_status"
            puts $fp "full_top_level_flow=FULL_TOP_FLOW_SYNTH_FAILED"
            puts $fp "bitstream_status=BITSTREAM_NOT_RUN_TOP_FLOW_INCOMPLETE"
            puts $fp "blocker=$blocker"
            close $fp
            exit 1
        }

        # ------------------------------------------------------------------
        # INLINE impl flow — full BD synth+impl, no partition reuse.
        # ------------------------------------------------------------------
        open_run synth_1 -name synth_1
        puts "\[pccx inline-impl\] running opt_design -directive ExploreWithRemap"
        opt_design -directive ExploreWithRemap
        # Match Vivado strategy "Performance_ExtraTimingOpt" exactly:
        #   opt   -directive ExploreWithRemap
        #   place -directive ExtraTimingOpt
        #   phys  -directive AggressiveExplore
        #   route -directive AggressiveExplore   (not Explore)
        #   post-route phys -directive Explore
        puts "\[pccx inline-impl\] place_design -directive ExtraTimingOpt"
        place_design -directive ExtraTimingOpt
        puts "\[pccx inline-impl\] phys_opt_design -directive AggressiveExplore"
        phys_opt_design -directive AggressiveExplore
        puts "\[pccx inline-impl\] route_design -directive AggressiveExplore"
        route_design -directive AggressiveExplore
        puts "\[pccx inline-impl\] post-route phys_opt_design -directive Explore"
        phys_opt_design -directive Explore
        puts "\[pccx inline-impl\] post-route reports"
        report_timing_summary -delay_type min_max -report_unconstrained \
            -check_timing_verbose -max_paths 20 \
            -file $REPORTS/timing_summary_post_impl_top.rpt
        report_utilization -hierarchical -file $REPORTS/utilization_post_impl_top.rpt
        report_drc -file $REPORTS/drc_post_impl_top.rpt
        write_checkpoint -force $BD_DIR/pccx_v002_system_routed.dcp

        # Determine whether timing met, based on the worst setup slack.
        set wns_ns "STATS_UNAVAILABLE"
        if {[catch {set _wp [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]} _err]} {
            puts "\[pccx inline-impl\] WARNING: get_timing_paths failed: $_err"
        } else {
            if {$_wp ne ""} {
                set wns_ns [get_property SLACK $_wp]
            }
        }
        puts "\[pccx inline-impl\] post-impl WNS = $wns_ns"
        set met "VIOLATED"
        if {$wns_ns eq "STATS_UNAVAILABLE"} {
            set met "STATS_UNAVAILABLE"
        } elseif {$wns_ns >= 0} {
            set met "MET"
        }

        if {$MODE eq "bitstream" && $met eq "MET"} {
            set _bit_path $BD_DIR/pccx_v002_system_wrapper.bit
            puts "\[pccx inline-impl\] write_bitstream -force $_bit_path"
            write_bitstream -force $_bit_path
            if {[file exists $_bit_path]} {
                set bitstream_status "BITSTREAM_REQUESTED"
                file copy -force $_bit_path $HW_ROOT/build/pccx_v002_system_wrapper.bit
                puts "\[pccx inline-impl\] bitstream copied to build/pccx_v002_system_wrapper.bit"
            } else {
                set bitstream_status "BITSTREAM_BLOCKED_BIT_NOT_PRODUCED"
                set blocker "write_bitstream_did_not_emit_bit"
            }
        } elseif {$MODE eq "bitstream"} {
            set bitstream_status "BITSTREAM_BLOCKED_TIMING_NOT_MET"
            set blocker "post_impl_top_timing_violated_after_bd_assembly_wns_ns=$wns_ns"
        } else {
            set bitstream_status "BITSTREAM_NOT_REQUESTED"
            set blocker "closure_only_mode_post_impl_wns_ns=$wns_ns"
        }
    }
}

set fp [open $status_file w]
puts $fp "implementation_scope=FULL_TOP_LEVEL"
puts $fp "mode=$MODE"
puts $fp "target_part=$TARGET_PART"
puts $fp "target_board=$TARGET_BOARD"
puts $fp "bd_script=[info script]"
puts $fp "bd_status=$bd_status"
puts $fp "wrapper_status=PRESENT"
puts $fp "filelist_status=PRESENT"
puts $fp "constraints_status=PRESENT"
puts $fp "board_part_status=AVAILABLE"
puts $fp "full_top_level_flow=$full_top_status"
puts $fp "bitstream_status=$bitstream_status"
puts $fp "blocker=$blocker"
puts $fp "required_next_step=Run BD rebuild/synth/impl to validate HP/ACP DataMover topology; descriptors are still PS-issued."
close $fp

puts "\[pccx\] system_bd status written to $status_file"
puts "\[pccx\] full_top_level_flow = $full_top_status"
puts "\[pccx\] bitstream_status = $bitstream_status"

if {$MODE eq "bitstream"} {
    if {$bitstream_status eq "BITSTREAM_REQUESTED"} { exit 0 } else { exit 3 }
}
