# Compile list for the kv260 integration build that consumes pccx-v002 via the third_party submodule.
#
# Ordering is owned by the reusable v002 package filelist. Paths follow the
# existing hw/vivado/filelist.f convention: entries are resolved from hw/.
#
# kv260-only RTL retained outside the pccx-v002 manifest:
#   vivado/npu_core_wrapper.sv

-f ../third_party/pccx-v002/LLM/scripts/filelist.f

# ===| KV260 packaging shim |==================================================
vivado/npu_core_wrapper.sv
